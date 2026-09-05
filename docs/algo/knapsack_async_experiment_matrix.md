# Knapsack Async RL Experiment Matrix

这份文档按两个维度整理最近几天的 knapsack RL 实验：

- prompt 采样方案：决定先抽哪些题。
- rollout n 分配方案：决定每道题生成多少个 response。

默认公共配置如无特别说明：

- model: `Qwen2.5-Math-1.5B`
- data: DAPO raw knapsack train file
- train trigger: `fixed_samples`
- batch size: `bs=128`
- nominal rollout n: `n=8`
- max rollout n: `16`
- rejection sampling: `false`
- advantage: `grpo`
- eval interval: `20`

## Prompt 采样方案

### sequential

按数据集顺序循环取 prompt。这个方案不主动偏向任何 pass rate 区间，适合作为最干净的 baseline。

### medium

按历史 pass rate 计算权重：

```text
weight = 4 * p * (1 - p)
```

因此 `p≈0.5` 的题权重最高，`p≈0` 或 `p≈1` 的题权重最低。它偏向不确定性最高的题。

### productive_mixed_plateau

先根据历史 pass rate 和 `medium_focus_plateau` 对应的 n，估计在该 n 下 rollout 出有正有负的概率：

```text
mixed_prob = 1 - p^n - (1-p)^n
```

然后使用两档权重：

```text
mixed_prob >= threshold -> weight = 1
otherwise               -> weight = 1/3
```

这个方案不是连续偏向 `p=0.5`，而是优先抽“在对应 n 下大概率能产生正负混合”的 prompt。

### medium_to_productive_mixed

prompt 采样从普通 `medium` 逐渐退火到 two-weight productive mixed。

```text
step = 0:
    weight = 4 * p * (1-p)

step = 400:
    if mixed_prob >= 0.8:
        weight = 1
    else:
        weight = 0.2
```

中间线性插值：

```text
weight = (1 - t) * medium_weight + t * productive_mixed_weight
t = step / 400
```

400 step 后，同一档内部近似 weighted uniform，不再继续强烈偏向 `p≈0.5`。

## Rollout n 分配方案

### fixed n

每个 prompt 固定生成 `n` 个 response，例如 `n=8`。不会根据历史 pass rate 或当前 rollout 结果改变预算。

### has positive and negative

持续 rollout，直到累计结果里至少有一个正样本和一个负样本，或者达到 `max_n`。

```text
stop if num_positive >= 1 and num_negative >= 1
else continue until max_n
```

### continue_on_mixed_chunk

每次生成一个 chunk，如果最近这个 chunk 同时有正有负，就继续 rollout；如果最近 chunk 全正或全负，就停止。

以 `chunk=4` 为例：

```text
chunk mixed      -> continue
chunk all pos/neg -> stop
max_n=16         -> hard stop
```

它看的是最近一个 chunk，不是累计是否有正有负。

### preallocated medium_focus

rollout 前根据历史 pass rate 一次性决定 target n：

```text
n = min_n + 4 * p * (1-p) * (max_n - min_n)
```

`p≈0.5` 时 n 最大，`p≈0/1` 时 n 最小。决定后中途不会根据当前 rollout 结果改变。

### preallocated productive_curve

rollout 前根据历史 pass rate 一次性决定 target n：

```text
p < 0.5:
    n 从 min_n 增加到 max_n

0.5 <= p <= 0.8:
    n = max_n

0.8 < p < 0.95:
    n 从 max_n 降到 min_n

p >= 0.95:
    n = min_n
```

决定后中途不变。

### online adaptive_chunk_mixed: uncertainty / medium_focus

每个 chunk 返回后，更新当前 prompt 的 `p_hat`，再重新计算 target n。

```text
w = min(1, current_rollout_count / 8)
p_hat = (1-w) * historical_pass_rate + w * current_rollout_pass_rate
```

target n 使用 medium focus：

```text
n = min_n + 4 * p_hat * (1-p_hat) * (max_n - min_n)
```

这是 online medium focus。`p_hat≈0.5` 给最大 n，越接近 0 或 1 给越小 n。

### online adaptive_chunk_mixed: productive_curve

同样在线更新 `p_hat`，但 target n 使用 productive curve：

```text
p_hat < 0.5:
    p_hat 越接近 0.5，n 越大

0.5 <= p_hat <= 0.8:
    n = max_n

0.8 < p_hat < 0.95:
    p_hat 越接近 0.95，n 越小

p_hat >= 0.95:
    n = min_n
```

这个方案认为 `0.5~0.8` 的 prompt 都属于 productive 区域，整段给满 budget。

### online adaptive_chunk_mixed: medium_focus_plateau

同样在线更新 `p_hat`，但 target n 是 medium focus 加一个平台：

```text
if 0.5 < p_hat < 0.8:
    n = max_n
else:
    n = min_n + 4 * p_hat * (1-p_hat) * (max_n - min_n)
```

它保留 medium focus 的两侧形状，但把 `0.5~0.8` 强行拉满。

### floor to chunk multiple

部分实验会把算出来的 target n 向下取整到 chunk size 的倍数。

以 `chunk=2` 为例：

```text
15 -> 14
13 -> 12
9  -> 8
```

因此实际 target n 只会是：

```text
2, 4, 6, 8, 10, 12, 14, 16
```

## Experiment Table

| # | Experiment name | Prompt sampling | N allocation | Online? | Chunk | Key idea | 性能 | 分析 |
|---:|---|---|---|---|---:|---|---|---|
| 1 | `seq_fixedn8_chunk4_fs1024_stale10_grpo_norej_epoch10_20260618` | `sequential` | fixed n=8 | No | 4 | 顺序 baseline，每题固定 8 个 response |  |  |
| 2 | `seq_continue_on_mixed_chunk4_fs1024_stale10_grpo_norej_epoch10_20260620` | `sequential` | continue_on_mixed_chunk | Yes | 4 | chunk mixed 就继续，全正/全负就停，最多 16 |  |  |
| 3 | `seq_adaptive_chunk_mixed_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_fresh_20260621_151045` | `sequential` | early adaptive_chunk_mixed / productive-region | Yes | 2 | 早期 online productive 区域方案，`0.5~0.8` 倾向大 n |  |  |
| 4 | `seq_adaptive_chunk_mixed_uncertainty_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_fresh_20260621_232235` | `sequential` | uncertainty / medium_focus | Yes | 2 | 按 `p_hat(1-p_hat)` 分配 n，`p≈0.5` 最大 |  |  |
| 5 | `seq_adaptive_chunk_mixed_productive_curve_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_fresh_20260622_111751` | `sequential` | productive_curve | Yes | 2 | 明确版 productive curve，`0.5~0.8` 给 max n |  |  |
| 6 | `priority_medium_adaptive_chunk_mixed_productive_curve_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_resp2k_20260622_2007` | `medium` | productive_curve | Yes | 2 | prompt 采样偏向 `p≈0.5`，prompt 内用 productive curve |  |  |
| 7 | `seq_productive_curve_floor_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_resp2k_20260623_0100` | `sequential` | preallocated productive_curve + floor | No | 2 | rollout 前用历史 p 决定 n，中途不改，n 向下对齐 chunk |  |  |
| 8 | `seq_adaptive_chunk_mixed_floor_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_resp2k_20260623_0100` | `sequential` | adaptive_chunk_mixed + floor | Yes | 2 | online 动态 n，并把 target n 向下对齐 chunk |  |  |
| 9 | `seq_adaptive_chunk_mixed_medium_focus_plateau_chunk2_fs1024_stale10_grpo_norej_ep10_eval20_resp2k_20260624_0028` | `sequential` | medium_focus_plateau | Yes | 2 | medium focus 两侧曲线，`0.5~0.8` 平台给 max n |  |  |
| 10 | `prod_mix_plateau_mfp_c2_ep10_0624` | `productive_mixed_plateau` | medium_focus_plateau | Yes | 2 | 两档 prompt 采样，prompt 内用 online medium_focus_plateau |  |  |
| 11 | `seq_omf_c2_ep10_0624r` | `sequential` | online medium_focus | Yes | 2 | 最简单 online medium focus，`n=2+4p_hat(1-p_hat)(16-2)` |  |  |
| 12 | `m2pm_omf_c2_ep10_0624r` | `medium_to_productive_mixed` | online medium_focus | Yes | 2 | prompt 采样从 medium 退火到 two-weight productive mixed，prompt 内用 online medium_focus |  |  |

## Suggested Comparisons

| 对比组 | 实验 | 主要看什么 | 性能 | 分析 |
|---|---|---|---|---|
| fixed n baseline | #1 vs adaptive 系列 | 动态 n 是否提高有效样本效率 |  |  |
| continue chunk | #2 vs #5/#9/#11 | 单看 chunk mixed 停止是否过于激进或过慢 |  |  |
| uncertainty vs productive | #4 vs #5 | `p≈0.5` 峰值策略 vs `0.5~0.8` productive 平台 |  |  |
| prompt priority | #5 vs #6 | 相同 productive_curve 下，medium prompt 采样是否有效 |  |  |
| preallocated vs online | #7 vs #8 | rollout 前定 n 和边 rollout 边改 n 的差异 |  |  |
| plateau ablation | #9 vs #11 | `0.5~0.8` 平台是否优于纯 medium focus |  |  |
| two-weight prompt sampling | #9 vs #10 | productive_mixed_plateau 采样是否优于 sequential |  |  |
| annealed prompt sampling | #11 vs #12 | medium-to-productive-mixed 退火采样是否优于 sequential |  |  |

## Metrics To Fill

建议每个实验至少填这些：

- validation performance: AIME24/AIME25/AMC/MATH 等。
- effective sample speed: 有效样本数 / wall-clock time。
- nonzero sample ratio: 有非零 advantage 的 sample 占比。
- mixed prompt ratio: batch 内有正有负的 prompt 占比。
- avg rollout n: 平均每个 prompt rollout 数。
- hit max ratio: 达到 max_n 的 prompt 比例。
- stale dropped ratio: 因 staleness 被过滤的样本比例。
- step time: 每个 train step 用时。
