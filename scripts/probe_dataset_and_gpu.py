"""Compare pass_rate distributions of the candidate DPSK-1.5B training sets."""
import os

import pandas as pd
import ray

D = "/taijifs_zw35/r2/ziniuli/processed_dataset/dapo/"
FILES = [
    ("baseline  trunc16384-v0", D + "train_deepscaler_format_evaluated_dpsk1.5b_n8_trunc16384-v0.parquet"),
    ("          trunc16384-v0_avglen_le8192",
     D + "train_deepscaler_format_evaluated_dpsk1.5b_n8_trunc16384-v0_avglen_le8192.parquet"),
    ("          trunc32768-v0", D + "train_deepscaler_format_evaluated_dpsk1.5b_n8_trunc32768-v0.parquet"),
    ("current   trunc32768-v0_clip001_099",
     D + "train_deepscaler_format_evaluated_dpsk1.5b_n8_trunc32768-v0_clip001_099.parquet"),
]
BINS = [0.0, 0.001, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.999, 1.001]

stats = {}
uidsets = {}
for label, path in FILES:
    if not os.path.exists(path):
        print("%s: 缺失" % label, flush=True)
        continue
    df = pd.read_parquet(path)
    ei = df["extra_info"]
    pr = ei.map(lambda d: float(d.get("pass_rate", float("nan"))))
    raw = ei.map(lambda d: float(d.get("raw_pass_rate", d.get("pass_rate", float("nan")))))
    ln = ei.map(lambda d: float(d.get("avg_length", float("nan"))))
    ov = ei.map(lambda d: float(d.get("overlong_ratio", float("nan"))))
    uid = ei.map(lambda d: d.get("prompt_uid"))
    uidsets[label.strip()] = set(uid)
    stats[label.strip()] = dict(
        n=len(df), pr=pr, raw=raw, ln=ln, ov=ov,
        prm=pr.mean(), rawm=raw.mean(), lnm=ln.mean(), ovm=ov.mean())
    print("\n=== %s ===" % label, flush=True)
    print("  行数 %d | pass_rate 均值 %.4f | raw 均值 %.4f | avg_length 均值 %.0f | overlong 均值 %.4f" % (
        len(df), pr.mean(), raw.mean(), ln.mean(), ov.mean()), flush=True)
    print("  pass_rate 分布:", flush=True)
    for i in range(len(BINS) - 1):
        m = (pr >= BINS[i]) & (pr < BINS[i + 1])
        n = int(m.sum())
        if n == 0:
            continue
        print("    [%.3f,%.3f)  %6d  %5.2f%%   平均长度 %6.0f" % (
            BINS[i], BINS[i + 1], n, 100.0 * n / len(df), ln[m].mean()), flush=True)
    ext = int(((pr <= 0.02) | (pr >= 0.98)).sum())
    print("  两端(<=0.02 或 >=0.98)占比 %.2f%%" % (100.0 * ext / len(df)), flush=True)
    print("  avg_length > 16384 的占比 %.2f%%（16K 预算下大概率做不完）" % (
        100.0 * float((ln > 16384).sum()) / len(df)), flush=True)

print("\n=== 集合关系 ===", flush=True)
ks = list(uidsets)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        a, b = uidsets[ks[i]], uidsets[ks[j]]
        print("  %s ∩ %s = %d  (前者 %d, 后者 %d, 前者被包含 %s)" % (
            ks[i][:34], ks[j][:34], len(a & b), len(a), len(b), a <= b), flush=True)

print("\n=== 集群 GPU ===", flush=True)
try:
    ray.init(address="auto", ignore_reinit_error=True)
    print("  总 GPU %s / 空闲 GPU %s" % (
        ray.cluster_resources().get("GPU"), ray.available_resources().get("GPU")), flush=True)
except Exception as exc:
    print("  查询失败 %r" % (exc,), flush=True)
