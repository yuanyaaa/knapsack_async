# Knapsack RL: Offline Medium-Focus Budget Compensation

本文整理当前的补偿版动态 rollout 方案，对应 stop rule:

```bash
STOP_RULE=offline_medium_focus_budget_compensate
```

核心目标是：先根据历史 pass rate 预分配每个 prompt 的 rollout 数 `n`，如果某个 prompt 最终全正或全负、没有有效 GRPO 信号，则在预算允许时追加 rollout，尽量把它变成有正有负的有效 group。

## 1. 初始动态 n

初始 `n` 使用 offline medium-focus，只依赖 prompt 的历史正确率 `p`，不看当前 rollout 结果：

```text
n_raw = n_min + 4 * p * (1 - p) * (n_max - n_min)
```

当前常用配置：

```text
n_min = 2
n_max = 16
```

offline initial n 是一次性按历史 pass rate 估计出来的，不要求和 rollout 请求粒度对齐，
因此直接向下取整到整数：

```text
n = floor(n_raw)
```

典型分配如下：

| pass rate p | initial n |
|---:|---:|
| 0.0 | 2 |
| 0.1 | 7 |
| 0.2 | 10 |
| 0.3 | 13 |
| 0.4 | 15 |
| 0.5 | 16 |
| 0.6 | 15 |
| 0.7 | 13 |
| 0.8 | 10 |
| 0.9 | 7 |
| 1.0 | 2 |

也就是说，`p=0.5` 附近会拿最多 rollout，因为最可能产生正负混合样本。

注意：`ROLLOUT_CHUNK_SIZE=2` 只影响每次向 rollout engine 请求多少条 response。若
`target_n` 是奇数，最后一次请求会自动只请求剩余的 1 条，因此不会强制把 initial n
对齐到偶数。

## 2. 什么时候补偿

prompt 完成初始 `n` 个 rollout 后，如果已经有正有负，就直接进入训练，不需要补偿。

只有以下两种情况才考虑补偿：

```text
全正：num_positive == num_finished
全负：num_positive == 0
```

这两种 group 在 GRPO 里通常没有有效 advantage 差异，因此补偿的目标是追加一些 response，让 group 有机会变成有正有负。

## 3. 滑动窗口预算

补偿不能无上限追加，否则总 rollout 成本会不可控。当前版本使用最近窗口预算，而不是全历史预算。

配置：

```bash
COMPENSATION_BUDGET_BASELINE_N=8
COMPENSATION_BUDGET_WINDOW_SIZE=512
```

含义是：只看最近 512 个 prompt，判断这 512 个 prompt 的平均 committed rollout 是否低于固定 `n=8` baseline。

窗口内维护：

```text
window_baseline_total = recent_prompt_count * 8
window_committed_total = 最近窗口中已经承诺生成的 rollout 数
window_available = max(0, window_baseline_total - window_committed_total)
```

补偿决策只使用：

```text
window_available
```

这样做的原因：

- 全历史 budget 会被早期 priority sampling 的高 `n` 负债长期锁死。
- 纯 non-negative bank 会忽略高 `n` prompt 的超支，约束太松。
- 滑动窗口只约束最近一段采样行为，更适合 priority sampling 分布随训练变化的场景。

## 4. 补偿 ROI

假设某个 prompt 初始完成了 `k` 个 rollout，并且全正或全负。现在考虑额外补偿 `extra` 个 rollout。

如果当前是全正，追加 `extra` 后至少出现一个负样本的概率为：

```text
success_prob = 1 - p^extra
```

如果当前是全负，追加 `extra` 后至少出现一个正样本的概率为：

```text
success_prob = 1 - (1 - p)^extra
```

补偿收益定义为：从无梯度 group 变成有效 mixed group 后，整个 group 的样本都变得有用。因此：

```text
expected_gain = (k + extra) * success_prob
ROI = expected_gain / extra
```

当前实验使用：

```bash
COMPENSATION_ROI_THRESHOLD=1.0
```

只有当：

```text
ROI > 1.0
```

才允许补偿。

## 5. 激进补偿策略

补偿不是固定 `+2`，而是枚举多个候选。由于补偿是一次性追加，候选不需要和初始 rollout 的 `chunk_size=2` 对齐，当前推荐用 `COMPENSATION_CHUNK_SIZE=1` 允许单数补偿：

```text
extra = 1, 2, 3, ..., min(n_max - k, window_available)
```

然后选择满足 `ROI > threshold` 的最大 `extra`。

伪代码：

```python
best_extra = 0
unit = compensation_chunk_size  # 推荐为 1
for extra in range(unit, max_extra + 1, unit):
    if all_positive:
        success_prob = 1 - p ** extra
    else:
        success_prob = 1 - (1 - p) ** extra

    roi = (k + extra) * success_prob / extra

    if roi > roi_threshold:
        best_extra = extra

if best_extra > 0:
    target_n = k + best_extra
    window_committed_total += best_extra
```

最终仍然满足：

```text
target_n <= n_max
```

在 `n_min=2, n_max=16, COMPENSATION_CHUNK_SIZE=1, ROI threshold=1.0`，且窗口预算
足够的情况下，按典型 pass rate 计算，补偿上限如下。

表中 `ROI峰值 extra/ROI` 表示单看 ROI 最大时对应的补偿数；`实际 extra/ROI` 表示当前
aggressive 策略会选择的最大补偿数，也就是满足 `ROI > threshold` 的最大 `extra`。
因此实际策略通常会比 ROI 峰值补得更多，以换取更高的 mixed group 成功概率。

| p | initial k | max extra cap | 全正 ROI峰值 extra/ROI | 全正实际 extra/ROI | 全负 ROI峰值 extra/ROI | 全负实际 extra/ROI |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 2 | 14 | 1/3.00 | 14/1.14 | - | - |
| 0.1 | 7 | 9 | 1/7.20 | 9/1.78 | 9/1.09 | 9/1.09 |
| 0.2 | 10 | 6 | 1/8.80 | 6/2.67 | 1/2.20 | 6/1.97 |
| 0.3 | 13 | 3 | 1/9.80 | 3/5.19 | 1/4.20 | 3/3.50 |
| 0.4 | 15 | 1 | 1/9.60 | 1/9.60 | 1/6.40 | 1/6.40 |
| 0.5 | 16 | 0 | - | - | - | - |
| 0.6 | 15 | 1 | 1/6.40 | 1/6.40 | 1/9.60 | 1/9.60 |
| 0.7 | 13 | 3 | 1/4.20 | 3/3.50 | 1/9.80 | 3/5.19 |
| 0.8 | 10 | 6 | 1/2.20 | 6/1.97 | 1/8.80 | 6/2.67 |
| 0.9 | 7 | 9 | 9/1.09 | 9/1.09 | 1/7.20 | 9/1.78 |
| 1.0 | 2 | 14 | - | - | 1/3.00 | 14/1.14 |

这里的 `max extra cap = n_max - initial k`，实际运行时还会再受 `window_available` 限制。

## 6. Prompt Sampling

当前 priority 版本使用：

```bash
SAMPLE_STRATEGY=lower_mid
```

对应权重：

```text
weight(p) = p * (1 - p)^2
```

它的峰值在 `p=1/3` 附近，语义是偏向 lower-mid 难度：不是最难，也不是 `p=0.5` 的标准 medium，而是更偏向“有一定概率做对但整体偏难”的 prompt。

## 7. 当前推荐配置

```bash
SAMPLE_STRATEGY=lower_mid
STOP_RULE=offline_medium_focus_budget_compensate
ADAPTIVE_ROLLOUT_N_STRATEGY=medium_focus

STOP_RULE_MIN_N=2
STOP_RULE_MAX_N=16
ROLLOUT_CHUNK_SIZE=2

COMPENSATION_CHUNK_SIZE=1
COMPENSATION_ROI_THRESHOLD=1.0
COMPENSATION_BUDGET_BASELINE_N=8
COMPENSATION_BUDGET_WINDOW_SIZE=512

BATCH_SIZE=128
N_RESP_PER_PROMPT=8
TRAIN_PROMPT_MINI_BSZ=128
TRAIN_TRIGGER=fixed_samples
REJECTION_SAMPLING=false
ADV_ESTIMATOR=grpo
STALENESS_THRESHOLD=10
```

## 8. 需要重点看的指标

预算相关：

```text
fully_async/budget/window_size
fully_async/budget/window_count
fully_async/budget/window_baseline_total
fully_async/budget/window_committed_total
fully_async/budget/window_available
fully_async/budget/window_available_raw
fully_async/budget/global_available
```

补偿相关：

```text
fully_async/compensation/prompts_total
fully_async/compensation/samples_total
fully_async/compensation/candidates_total
fully_async/compensation/rejected_roi_total
fully_async/compensation/success_ratio
fully_async/compensation/final_acc_mean
fully_async/compensation/roi_mean
fully_async/compensation/success_prob_mean
fully_async/compensation/extra_n_mean
```

每步补偿情况：

```text
fully_async/compensation/prompts_per_step
fully_async/compensation/samples_per_step
fully_async/compensation/success_ratio_per_step
fully_async/compensation/final_acc_mean_per_step
fully_async/compensation/roi_mean_per_step
```

## 9. 和旧方案的区别

### 固定 +2 补偿

旧的小补偿版本只在小 `n` 且全正/全负时追加 `+2`。它实现简单，但补偿力度较弱，很多 prompt 仍然无法变成有正有负。

### 全历史 budget

全历史 budget 使用：

```text
global_available = total_prompt_count * 8 - total_committed
```

这个在 priority sampling 下容易出问题：前期如果采到很多高 `n` prompt，`global_available` 会长期为负，后期即使采到低 `n` prompt 也很难补偿。

### 当前滑动窗口 budget

当前方案只看最近 512 个 prompt：

```text
window_available = recent_baseline - recent_committed
```

它既会惩罚窗口内的高 `n` 超支，也不会让早期负债永久影响后期，是目前比较合理的折中。
