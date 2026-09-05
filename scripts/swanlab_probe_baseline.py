"""Pull config + metric series for the three DPSK-1.5B runs from SwanLab."""
import json
import os

import swanlab

swanlab.login(api_key=os.environ.get("USER_SWANLAB_API_KEY"), save=False)
api = swanlab.OpenApi()

WS = os.environ.get("SWANLAB_WORKSPACE", "X4")
PROJ = os.environ.get("PROJECT_NAME", "wangpengyuan_verl")

RUNS = [
    ("baseline_fixedn8", "hyh3x33lh9qkpp6yctcd4"),
    ("shuffle_adaptive", "bnixpmc66uuyoz644go1g"),
    ("lower_mid", "5widkdcdvsxdxvcwljg3l"),
]
SECRET = ("key", "token", "secret", "password", "wechat", "webhook")


def unwrap(obj):
    """Peel SwanLab's {code,data,errmsg} envelope and pydantic wrappers."""
    seen = 0
    while seen < 6:
        seen += 1
        if hasattr(obj, "model_dump"):
            try:
                obj = obj.model_dump()
                continue
            except Exception:
                pass
        if isinstance(obj, dict) and set(obj.keys()) & {"code", "errmsg"} and "data" in obj:
            obj = obj["data"]
            continue
        if not isinstance(obj, dict) and hasattr(obj, "data"):
            obj = obj.data
            continue
        break
    return obj


def flatten(d, pre=""):
    """Flatten SwanLab config; unwrap {'value':...,'desc':...} leaves and recurse into them."""
    out = {}
    if isinstance(d, str):
        try:
            d = json.loads(d)
        except Exception:
            return {pre: d} if pre else {}
    if isinstance(d, list):
        return {pre: json.dumps(d, ensure_ascii=False, sort_keys=True)} if pre else {}
    if not isinstance(d, dict):
        return {pre: d} if pre else {}
    for k, v in d.items():
        p = "%s.%s" % (pre, k) if pre else str(k)
        if isinstance(v, dict) and "value" in v and set(v.keys()) <= {
                "value", "desc", "sort", "type", "class"}:
            v = v["value"]
        if isinstance(v, (dict, list)) or (
                isinstance(v, str) and v[:1] in ("{", "[")):
            sub = flatten(v, p)
            out.update(sub if sub else {p: v})
        else:
            out[p] = v
    return out


KEYS = [] or [
    "val-core/overall/acc/mean@16", "val-core/overall/acc/maj@16/mean",
    "val-core/MATH/acc/mean@16", "val-core/AIME/acc/mean@16",
    "val-core/AIME2025/acc/mean@16", "val-core/AMC/acc/mean@16",
    "val-core/MINERVA/acc/mean@16", "val-core/OLYMPIAD_BENCH/acc/mean@16",
    "val-core/Idavidrein/gpqa/acc/mean@16",
    "batch_analysis/advantage/nonzero_adv_sample_ratio",
    "batch_analysis/rollout_acc/overall", "actor/grad_norm", "actor/entropy",
    "response_length/mean", "critic/score/mean",
]

configs = {}
first = True
for label, cuid in RUNS:
    print("=" * 100, flush=True)
    print("### %s" % label, flush=True)
    print("=" * 100, flush=True)

    exp = None
    for call in (
        lambda: api.get_experiment(PROJ, cuid, WS),
        lambda: api.get_experiment(project=PROJ, exp_id=cuid, username=WS),
    ):
        try:
            exp = unwrap(call())
            break
        except Exception as exc:
            print("  get_experiment err: %r" % (exc,), flush=True)

    prof = exp.get("profile") if isinstance(exp, dict) else None
    if first and isinstance(exp, dict):
        print("  [结构] experiment keys: %s" % sorted(exp.keys()), flush=True)
        if isinstance(prof, dict):
            print("  [结构] profile 顶层 key: %s" % sorted(prof.keys()), flush=True)

    raw_cfg = None
    if isinstance(prof, dict):
        for probe in ("config", "conf", "hyperparameters", "params"):
            if isinstance(prof.get(probe), dict):
                raw_cfg = prof[probe]
                break
        if raw_cfg is None:
            raw_cfg = prof
    cfg = flatten(raw_cfg)
    cfg = {k: v for k, v in cfg.items() if not any(s in k.lower() for s in SECRET)}
    configs[label] = cfg
    print("  config 展平后 %d 项" % len(cfg), flush=True)

    series = {}
    for mk in KEYS:
        r = None
        for call in (
            lambda: api.get_metrics(cuid, [mk]),
            lambda: api.get_metrics(cuid, mk),
        ):
            try:
                r = call()
                break
            except Exception as exc:
                err = exc
        if r is None:
            if first:
                print("  get_metrics err: %r" % (err,), flush=True)
            continue
        rows = []
        df = unwrap(r)
        if hasattr(df, "iterrows"):
            vc = mk if mk in df.columns else next(
                (c for c in df.columns if "timestamp" not in str(c).lower()), None)
            if first and mk == KEYS[0]:
                print("  [结构] DataFrame 列=%s index=%s value=%s" % (
                    [str(c) for c in df.columns], df.index.name, vc), flush=True)
            for idx, val in df[vc].items():
                try:
                    rows.append((int(idx), float(val)))
                except Exception:
                    continue
        else:
            d = df
            cand = d.get(mk) if isinstance(d, dict) and mk in d else d
            if isinstance(cand, dict):
                cand = cand.get("list", cand.get("items", []))
            try:
                for p in cand:
                    if isinstance(p, dict):
                        s = p.get("step", p.get("index"))
                        v = p.get("value", p.get("data"))
                    else:
                        s, v = getattr(p, "step", None), getattr(p, "value", None)
                    if s is not None and v is not None:
                        rows.append((int(s), float(v)))
            except Exception as exc:
                if first:
                    print("  parse err %s: %r" % (mk, exc), flush=True)
        if rows:
            rows.sort()
            series[mk] = rows
    first = False

    print("  --- 验证曲线（每 50 步）---", flush=True)
    for mk in KEYS:
        if not mk.startswith("val-") or mk not in series:
            continue
        pts = [r for r in series[mk] if r[0] % 50 == 0] or series[mk]
        print("    %-40s %s" % (mk[9:], " ".join("s%d=%.4f" % r for r in pts)), flush=True)
    print("  --- 训练指标 ---", flush=True)
    for mk in KEYS:
        if mk.startswith("val-") or mk not in series:
            continue
        v = [x[1] for x in series[mk]]
        e = [x[1] for x in series[mk] if x[0] <= 207]
        print("    %-45s n=%4d 末步=%4d 全均=%9.4f 前207均=%9.4f" % (
            mk, len(v), series[mk][-1][0], sum(v) / len(v),
            (sum(e) / len(e)) if e else float("nan")), flush=True)

print("\n" + "=" * 100, flush=True)
print("### 超参数逐项对比", flush=True)
print("=" * 100, flush=True)
b = configs.get("baseline_fixedn8", {})
l = configs.get("lower_mid", {})
s = configs.get("shuffle_adaptive", {})
allk = sorted(set(b) | set(l) | set(s))
print("%-3s %-56s | %-24s | %-24s | %s" % ("", "key", "baseline_fixedn8", "lower_mid", "shuf_adaptive"), flush=True)
nd = 0
for k in allk:
    bv, lv, sv = b.get(k, "<缺>"), l.get(k, "<缺>"), s.get(k, "<缺>")
    if str(bv) == str(lv) == str(sv):
        continue
    nd += 1
    print("%-3s %-56s | %-24s | %-24s | %s" % ("!!", k, str(bv)[:24], str(lv)[:24], str(sv)[:24]), flush=True)
print("\n共 %d 项 key，其中 %d 项三者不完全一致（上表），%d 项完全一致" % (len(allk), nd, len(allk) - nd), flush=True)

FULL = ["actor_rollout_ref.model.path", "data.train_files", "data.val_files",
        "data.max_response_length", "data.train_batch_size", "data.gen_batch_size",
        "actor_rollout_ref.actor.optim.lr", "actor_rollout_ref.actor.loss_agg_mode",
        "actor_rollout_ref.actor.ppo_mini_batch_size", "actor_rollout_ref.actor.ppo_epochs",
        "actor_rollout_ref.actor.use_kl_loss", "actor_rollout_ref.actor.kl_loss_coef",
        "actor_rollout_ref.actor.clip_ratio", "actor_rollout_ref.actor.entropy_coeff",
        "actor_rollout_ref.rollout.n", "actor_rollout_ref.rollout.temperature",
        "actor_rollout_ref.rollout.val_kwargs.n", "actor_rollout_ref.rollout.val_kwargs.temperature",
        "actor_rollout_ref.rollout.val_kwargs.max_response_length",
        "algorithm.adv_estimator", "algorithm.norm_adv_by_std_in_grpo",
        "algorithm.filter_groups.enable", "trainer.total_epochs",
        "async_training.rollout_config.required_samples_per_step",
        "async_training.rollout_config.medium_focus_peak_n",
        "async_training.rollout_config.min_rollout_n",
        "async_training.rollout_config.max_rollout_n",
        "async_training.rollout_config.staleness_threshold",
        "async_training.rollout_config.adaptive_chunk_prior_tau",
        "async_training.rollout_config.compensation_budget_baseline_n",
        "async_training.rollout_config.compensation_budget_window_size",
        "async_training.max_staleness", "rollout.total_rollout_steps"]
print("\n" + "=" * 100, flush=True)
print("### 关键项完整值", flush=True)
print("=" * 100, flush=True)
for k in FULL:
    bv, lv, sv = b.get(k, "<缺>"), l.get(k, "<缺>"), s.get(k, "<缺>")
    same = "相同" if str(bv) == str(lv) == str(sv) else "不同"
    print("[%s] %s" % (same, k), flush=True)
    print("      baseline : %s" % (bv,), flush=True)
    if str(bv) != str(lv):
        print("      lower_mid: %s" % (lv,), flush=True)
    if str(sv) != str(lv):
        print("      shuf_adp : %s" % (sv,), flush=True)
