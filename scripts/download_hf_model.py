#!/usr/bin/env python3
"""Download a HuggingFace model onto the shared cluster filesystem via the corp proxy.

The cluster nodes have no direct egress, so http(s)_proxy must point at star-proxy.
Downloads land in a sibling ``.tmp`` directory and are renamed into place only after
the shard list is verified, so an interrupted run can never leave behind a directory
that looks like a complete model.
"""

import argparse
import json
import os
import shutil
import sys
import time

PROXY = "http://star-proxy.oa.com:3128"


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"


def dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def verify(path):
    """Return (ok, message). Checks config, tokenizer and every indexed shard."""
    required = ["config.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(path, f))]
    if missing:
        return False, f"缺少必要文件: {missing}"

    if not any(os.path.exists(os.path.join(path, f)) for f in ("tokenizer.json", "tokenizer_config.json")):
        return False, "缺少 tokenizer 文件"

    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as fh:
            index = json.load(fh)
        weight_map = index.get("weight_map", {})
        shards = sorted(set(weight_map.values()))
        absent = [s for s in shards if not os.path.exists(os.path.join(path, s))]
        if absent:
            return False, f"索引声明的分片缺失: {absent}"

        # Comparing bytes against metadata.total_size does not work: models with tied
        # embeddings count the shared tensor twice there while storing it once on disk.
        # Reading each shard's safetensors header instead proves every indexed tensor
        # actually arrived, which is what we care about.
        for shard in shards:
            shard_path = os.path.join(path, shard)
            with open(shard_path, "rb") as fh:
                header_len = int.from_bytes(fh.read(8), "little")
                if header_len <= 0 or header_len > os.path.getsize(shard_path):
                    return False, f"{shard} 头部长度非法 ({header_len}),文件可能被截断"
                header = json.loads(fh.read(header_len).decode("utf-8"))
            present = {k for k in header if k != "__metadata__"}
            expected_tensors = {t for t, s in weight_map.items() if s == shard}
            lost = expected_tensors - present
            if lost:
                return False, f"{shard} 缺少张量 {sorted(lost)[:5]} (共 {len(lost)} 个)"
            # A truncated tail is invisible to the header, so confirm the declared data
            # region fits inside the file.
            end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
            if 8 + header_len + end > os.path.getsize(shard_path):
                return False, f"{shard} 数据区超出文件大小,下载不完整"

        return True, f"{len(shards)} 个分片、{len(weight_map)} 个张量齐全"

    if os.path.exists(os.path.join(path, "model.safetensors")):
        return True, "单分片模型"
    return False, "既无 index.json 也无 model.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--dest", required=True, help="最终模型目录")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--endpoint", default="https://huggingface.co")
    ap.add_argument("--force", action="store_true", help="目标已存在且完整时也重新下载")
    args = ap.parse_args()

    os.environ["http_proxy"] = os.environ.get("http_proxy") or PROXY
    os.environ["https_proxy"] = os.environ.get("https_proxy") or PROXY
    os.environ["HF_ENDPOINT"] = args.endpoint
    # hf_transfer is not installed on these images and silently breaks the download path.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    print(f"repo    = {args.repo_id}", flush=True)
    print(f"dest    = {args.dest}", flush=True)
    print(f"endpoint= {args.endpoint}  proxy={os.environ['https_proxy']}", flush=True)

    if os.path.exists(args.dest) and not args.force:
        ok, msg = verify(args.dest)
        if ok:
            print(f"目标已存在且完整({msg}, {human(dir_size(args.dest))}),跳过下载。", flush=True)
            return 0
        print(f"目标已存在但不完整({msg}),将重新下载。", flush=True)
        shutil.rmtree(args.dest)

    from huggingface_hub import snapshot_download

    tmp = args.dest.rstrip("/") + ".tmp"
    if os.path.exists(tmp):
        # snapshot_download keeps etag metadata under tmp/.cache and skips files that are
        # already complete, so reusing the directory makes a retry nearly free.
        print(f"复用已有临时目录 {tmp} ({human(dir_size(tmp))}),将断点续传", flush=True)
    os.makedirs(tmp, exist_ok=True)

    t0 = time.time()
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=tmp,
        max_workers=args.workers,
        # The repo carries no useful non-weight extras; skipping git metadata keeps the
        # tree clean for from_pretrained.
        ignore_patterns=[".gitattributes"],
    )
    elapsed = time.time() - t0
    size = dir_size(tmp)
    print(f"下载完成: {human(size)}, 耗时 {elapsed/60:.1f} 分钟, 均速 {human(size/max(elapsed,1))}/s", flush=True)

    ok, msg = verify(tmp)
    if not ok:
        print(f"校验失败: {msg}", flush=True)
        return 1
    print(f"校验通过: {msg}", flush=True)

    # snapshot_download leaves its resume metadata inside local_dir; it is dead weight
    # once the download is complete and confuses anyone listing the model directory.
    cache_meta = os.path.join(tmp, ".cache")
    if os.path.isdir(cache_meta):
        shutil.rmtree(cache_meta)

    os.rename(tmp, args.dest)
    print(f"已就位: {args.dest}", flush=True)
    for f in sorted(os.listdir(args.dest)):
        p = os.path.join(args.dest, f)
        print(f"   {f:44s} {human(os.path.getsize(p)) if os.path.isfile(p) else '<dir>'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
