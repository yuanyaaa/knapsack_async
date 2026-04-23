
# Fully Async Policy — 指标（Metrics）说明文档

> 本文档梳理 `verl/experimental/fully_async_policy/` 在训练过程中上报到日志 / TensorBoard / WandB 的所有指标含义，并着重说明 **Rejection Sampling** 与 **Pad Samples** 这两类在完全异步链路中新引入的特有指标。
>
> - 所有 fully-async 专属指标统一以前缀 `fully_async/` 打点。
> - 所有 batch-level 训练信号分析指标统一以前缀 `batch_analysis/` 打点。
> - 代码入口主要位于：
>   - [detach_utils.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/detach_utils.py)（`assemble_batch_from_rollout_samples` / `compute_staleness_metrics` / `MetricsAggregator`）
>   - [fully_async_trainer.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/fully_async_trainer.py)（pad / rejection sampling 注入 `meta_info`）
>   - [utils/batch_metrics.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/utils/batch_metrics.py)（batch analysis 指标）

---

## 1. 指标命名空间总览

| 命名空间 | 产出模块 | 粒度 | 说明 |
|---|---|---|---|
| `fully_async/count/*` | Rollouter → Trainer | 累计量（`last` 聚合） | 样本/Prompt 的生成、丢弃、拒绝总数 |
| `fully_async/partial/*` | Trainer（batch 组装时） | batch 级 | Partial rollout（跨参数版本）统计 |
| `fully_async/staleness/*` | Trainer（batch 组装后） | batch 级 | Response / Prompt 级参数版本 staleness |
| `fully_async/prompt_buffer/*` | Trainer（batch 组装时） | batch 级 | Prompt buffer 侧的 pass_rate / 采样概率 / rollout_n 统计 |
| `fully_async/rejection_sampling/*` | Rollouter → Trainer | 累计量（`last` 聚合） | **Rejection Sampling 专属统计**（见 §4） |
| `fully_async/pad/*` | Trainer | batch 级 | **Pad Samples 专属统计**（见 §5） |
| `fully_async/purge/*` | Trainer（param sync 后） | 事件级（仅非零时上报） | **Queue 侧 stale item 清理统计**（见 §5.5） |
| `fully_async/rollouter/*` | Rollouter | 周期性 | Rollouter 侧活跃时间等运行状态 |
| `fully_async/trainer/*` | Trainer | step 级 | Trainer 侧空转率等 |
| `batch_analysis/difficulty/*` | `batch_metrics.py` | batch 级 | Prompt 难度分箱 |
| `batch_analysis/advantage/*` | `batch_metrics.py` | batch 级 | 非零 advantage prompt/sample 占比 |
| `batch_analysis/response_length/*` | `batch_metrics.py` | batch 级 | Correct / Incorrect 分组长度与 overlong 比例 |
| `batch_analysis/raw_acc/*` | `batch_metrics.py` | batch 级 | **Raw accuracy（含被拒绝样本）** |
| `batch_analysis/unweighted_acc/*` | `batch_metrics.py` | batch 级 | 重要性采样校正后的 accuracy（Priority Sampling 时启用） |

---

## 2. `fully_async/count/*` — 累计计数指标

在 `MetricsAggregator` 中这些 key 被归类为 `last` 聚合（每次 report 取最新值），代码见 [detach_utils.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/detach_utils.py) 的 `_init_aggregation_rules`。

| 指标 | 粒度 | 含义 |
|---|---|---|
| `fully_async/count/total_generated_prompts` | prompt | Rollouter 成功 emit 到 MessageQueue 的 prompt 累计数 |
| `fully_async/count/total_generated_samples` | sample | Rollouter 成功 emit 到 MessageQueue 的 sample（response）累计数 |
| `fully_async/count/stale_prompts` | prompt | 已生成但因超过 staleness 阈值被丢弃的 prompt 累计数 |
| `fully_async/count/stale_samples` | sample | 已生成但因超过 staleness 阈值被丢弃的 sample 累计数 |
| `fully_async/count/dropped_prompts` | prompt | 因 put queue 失败等原因被丢弃的 prompt 累计数 |
| `fully_async/count/dropped_samples` | sample | 因 put queue 失败等原因被丢弃的 sample 累计数 |
| `fully_async/count/rejected_prompts` | prompt | Rejection Sampling 丢弃的 prompt 累计数（见 §4） |
| `fully_async/count/rejected_samples` | sample | Rejection Sampling 丢弃的 sample 累计数（见 §4） |
| `fully_async/count/current_param_version` | scalar | Rollouter 当前持有的参数版本号 |

> 注意：`total_generated_*` 统计的是 **emit 成功** 的数量；如果 `rejection_sampling=True`，被 Rollouter 侧过滤的 prompt 不会进入该计数，会记入 `rejected_*`。

---

## 3. `fully_async/partial/*` — Partial Rollout 统计

Partial rollout 指一条 sample 在生成过程中跨越了至少一次参数更新（`min_global_steps != max_global_steps`）。

| 指标 | 含义 |
|---|---|
| `fully_async/partial/total_partial_num` | batch 中 `param_version_diff != 0` 的 sample 数 |
| `fully_async/partial/partial_ratio` | `total_partial_num / len(batch)` |
| `fully_async/partial/max_partial_span` | batch 内最大的 `|max_global_steps - min_global_steps|` |

---

## 4. `fully_async/rejection_sampling/*` — Rejection Sampling 指标 ⭐

### 4.1 机制简述

开启 `async_training.rollout_config.rejection_sampling=True` 后，**Rollouter 在 emit 到 MessageQueue 之前** 会丢弃两类无训练信号的 prompt：

- **`solve_all`**：该 prompt 下所有 response 全部答对（reward 全正），group-advantage 为 0。
- **`solve_none`**：该 prompt 下所有 response 全部答错（reward 全零/负），group-advantage 同样为 0。

仅当 prompt 产生了 **mixed outcome**（部分正确 / 部分错误）时才会被 accept，继续送往 trainer。

相关代码：
- 过滤点：`FullyAsyncRollouter._emit_prompt_as_rollout_sample`（[fully_async_rollouter.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/fully_async_rollouter.py) 约 L389–L416，维护 `self._rejection_stats`）
- 统计消费：`FullyAsyncTrainer` 组 batch 后异步调用 `rollouter.consume_rejection_stats.remote()` 拉取并清零统计（[fully_async_trainer.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/fully_async_trainer.py) L428–L477）。

> 设计选择：在 Rollouter 侧过滤而非 Trainer 侧，可避免被拒绝的 prompt 参与跨进程序列化与传输，显著节省通信开销。

### 4.2 Prompt-level 指标

| 指标 | 含义 |
|---|---|
| `fully_async/rejection_sampling/prompt/total_seen` | 自上次 consume 起 Rollouter 观察到的总 prompt 数（accepted + rejected） |
| `fully_async/rejection_sampling/prompt/accepted` | 通过筛选、已 emit 到 queue 的 prompt 数 |
| `fully_async/rejection_sampling/prompt/rejected_solve_all` | 被判为 **全对** 而丢弃的 prompt 数 |
| `fully_async/rejection_sampling/prompt/rejected_solve_none` | 被判为 **全错** 而丢弃的 prompt 数 |
| `fully_async/rejection_sampling/prompt/accept_ratio` | `accepted / total_seen`，反映当前难度分布与模型能力的匹配度 |
| `fully_async/rejection_sampling/prompt/solve_all_ratio` | `rejected_solve_all / (accepted + rejected_*)`，过高 → 数据过易 / 模型已饱和 |
| `fully_async/rejection_sampling/prompt/solve_none_ratio` | `rejected_solve_none / (accepted + rejected_*)`，过高 → 数据过难 / 模型未收敛 |

### 4.3 Sample-level 指标

Prompt 被拒绝时，同时会把该 prompt 下所有 response 计入 sample 维度：

| 指标 | 含义 |
|---|---|
| `fully_async/rejection_sampling/sample/total_seen` | 自上次 consume 起的总 response 数 |
| `fully_async/rejection_sampling/sample/accepted` | 通过筛选的 response 数 |
| `fully_async/rejection_sampling/sample/rejected_solve_all` | 因 solve_all 被丢弃的 response 数 |
| `fully_async/rejection_sampling/sample/rejected_solve_none` | 因 solve_none 被丢弃的 response 数 |
| `fully_async/rejection_sampling/sample/accept_ratio` | `accepted / total_seen` |
| `fully_async/rejection_sampling/sample/solve_all_ratio` | 同 Prompt 级，但以 sample 为单位 |
| `fully_async/rejection_sampling/sample/solve_none_ratio` | 同 Prompt 级，但以 sample 为单位 |

### 4.4 `batch_analysis/raw_acc/*` 与 `batch_analysis/unweighted_acc/*`

由于 Trainer 只看到 accepted 样本，直接统计 accuracy 会 **高估** 模型真实水平。为此 rollouter 会把每个 prompt 的原始 `(correct_count, total_count, sampling_prob)` 通过 `meta_info["rejection_sampling_reward_info"]` 一并上报，由 `compute_raw_acc_with_importance_sampling`（[utils/batch_metrics.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/utils/batch_metrics.py)）计算：

| 指标 | 含义 |
|---|---|
| `batch_analysis/raw_acc/mean` | 未过滤的 accuracy：`sum(correct) / sum(total)`，囊括被拒绝的 solve_all/solve_none |
| `batch_analysis/raw_acc/total_correct` | 所有 prompt 的正确 response 总数 |
| `batch_analysis/raw_acc/total_responses` | 所有 prompt 的 response 总数 |
| `batch_analysis/raw_acc/total_prompts` | 参与统计的 prompt 数 |
| `batch_analysis/unweighted_acc/mean` | **重要性采样修正**后的 accuracy，仅在 Priority Sampling 激活（存在非 uniform 的 `sampling_prob`）时出现。公式：`sum(acc_i / p_i) / sum(1 / p_i)`，用以抵消非均匀采样造成的偏差，逼近总体均匀采样下的 acc |

> `raw_pass_rate` 与 `pass_rate`（clip 后）区分的思想与此类似：前者反映真实分布，后者反映被过滤后进入训练的分布。

---

## 5. `fully_async/pad/*` — Pad Samples 指标 ⭐

### 5.1 为什么需要 Pad

在 fully-async 场景下，由于启用了 `fixed_samples` / `max_rollout_n` / `has_at_least_positive` 等动态 rollout 策略，单次组装出来的 batch **sample 数不定**，未必能被 `ppo_mini_batch_size (mbs)` 整除，也未必能被 actor `world_size` 整除。下游 `DataProto` 迭代器会 assert `total % world_size == 0`，因此需要 pad。

相关代码：[fully_async_trainer.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/fully_async_trainer.py) L385–L425。

### 5.2 Pad 策略

1. 计算 `divisor = lcm(mbs, world_size)`；
2. 若 `len(batch) % divisor != 0`，调用 `pad_dataproto_to_divisor` 复制已有行进行补齐；
3. **关键**：将 pad 行的 `response_mask` 强制置零，使其对 loss / 梯度无贡献（即"哑样本"）；
4. 整个过程中 list / ndarray 类 `meta_info` 会被临时摘除再恢复，避免 `DataProto.concat` 的等值断言冲突。

### 5.3 指标含义

| 指标 | 含义 |
|---|---|
| `fully_async/pad/pre_pad_sequences` | pad **之前** 的真实 sample 数（有效训练样本数） |
| `fully_async/pad/pad_size` | 本 step 实际补齐的哑样本条数；为 0 表示刚好整除无需 pad |
| `fully_async/pad/divisor` | 当前对齐使用的除数 `lcm(ppo_mini_batch_size, actor_world_size)` |

### 5.4 使用建议

- `pad_size / (pre_pad_sequences + pad_size)` 越高，越多算力被浪费在哑样本上 → 可考虑调大 `ppo_mini_batch_size` 的因子对齐，或收紧 `max_rollout_n` 浮动范围。
- `pad_size` 长期为 0 通常意味着采样数固定（`fixed_sequence` + 固定 N），属于正常状态。
- 由于 pad 行 `response_mask=0`，它们 **不会** 影响 `batch_analysis/*` 的 loss 相关指标，但 **可能** 影响一些按 `len(batch)` 直接除法的指标 —— 分析时应优先使用 `pre_pad_sequences` 作为分母。

---

### 5.5 `fully_async/purge/*` — Queue 侧 Stale Item 清理指标 ⭐

#### 5.5.1 机制简述

完全异步训练下，MessageQueue 中可能长期堆积着「生成时参数版本太旧」的样本。为防止下一个 training step 拿到 staleness 过大的数据，Trainer 在 **参数同步完成后立刻** 调用 `MessageQueue.purge_stale_samples(current_param_version, staleness_threshold)`，把 `staleness = current_param_version - start_version >= staleness_threshold` 的队列项整体移除。

关键代码：
- 队列端实现：`MessageQueue.purge_stale_samples`（[message_queue.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/message_queue.py) 约 L246–L304）。通过 sidecar deque `_start_versions` 无需反序列化即可判断 staleness，sentinel 项（`start_version == -1`）永不被 purge。
- 触发与打点：`FullyAsyncTrainer._fit_update_weights`（[fully_async_trainer.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/fully_async_trainer.py) 约 L914–L935），仅当 `self.staleness_threshold > 0` 时才执行，且仅当 `purged_prompts > 0` 时才写入 logger。上报 step 取当前 `self.current_param_version`。

> 时序要点：`purge` 发生在 `reset_staleness` 之后、下一次 `_fit_generate` 消费之前。由于 `current_param_version` 在 `_fit_update_local_step` 之前 **不会** 再增大，下一批被消费样本的 staleness 保证严格 `< staleness_threshold`。

#### 5.5.2 指标含义

| 指标 | 含义 |
|---|---|
| `fully_async/purge/purged_prompts` | 本次 purge 从队列中移除的 **queue item 数**（每个 item = 一个 prompt 携带的若干条 response） |
| `fully_async/purge/purged_samples` | 本次 purge 移除的 **response 条数**（`sum(n_samples)`），即真正被丢弃的训练样本量 |

> ⚠️ 注意：该指标 **不会周期性上报 0**。当 `staleness_threshold <= 0` 或本次没有任何样本命中 staleness 条件时，logger 不写入，TB 上会表现为「稀疏事件序列」。分析时应结合 `fully_async/staleness/response_staleness_max` 与 `fully_async/count/total_generated_samples` 等指标综合判断。

#### 5.5.3 与 `fully_async/count/stale_*` 的区别

`fully_async/count/stale_prompts` / `stale_samples` 记录的是 **Rollouter 侧**（生成端）因 `reset_staleness` 判定而丢弃的未入队样本；而 `fully_async/purge/*` 记录的是 **Queue 侧**（消费端入口）已经入队、在 param sync 之后才被判定为过期而清理掉的样本。两者互补，共同构成双层 staleness 防护：

```
Rollouter(生成端) -[reset_staleness 丢弃]-> stale_prompts/stale_samples
      │
      ▼ put_sample
MessageQueue(队列) -[purge_stale_samples 清理]-> purge/purged_*
      │
      ▼ get_batch
Trainer(消费端) ← 保证 staleness < S
```

---

### 5.6 MessageQueue Size 相关观测（非周期性 TB 指标）

> **澄清**：当前代码 **没有** 把 `fully_async/queue/size` 这类周期性队列水位指标自动打到 TB / WandB。下列字段是 `MessageQueue` 提供的 **可程序化查询** 的状态量，主要在日志 / 调试 / 对账场景使用，分析时可以从 stdout 或额外自定义打点中获取。

#### 5.6.1 `MessageQueue.get_statistics()` 返回字段

相关代码：[message_queue.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/message_queue.py) 约 L235–L245。

| 字段 | 单位 | 含义 |
|---|---|---|
| `queue_size` | queue item | 当前队列中 **item 数**（1 item = 1 prompt 的 rollout 结果） |
| `pending_samples` | response | 当前队列中累计的 **response 条数** = `sum(n_samples)`；变长 rollout 时 `queue_size * rollout.n` 不再准确，应以此为准 |
| `total_produced` | response | 队列实例生命周期内累计被 `put_sample` 成功入队的 sample 数 |
| `total_consumed` | queue item | 队列实例生命周期内累计被 `get_batch` 消费的 item 数 |
| `dropped_samples` | response | 因队列满（`len(queue) >= max_queue_size`）被丢弃的 sample 数；与 `fully_async/count/dropped_samples` 语义一致 |
| `max_queue_size` | queue item | 队列容量上限，初始化时通过 `MessageQueue.remote(config, max_queue_size)` 设置，见 [fully_async_main.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/fully_async_main.py) L144–L147 |

#### 5.6.2 `get_batch` 返回结构中的 queue 水位

Trainer 每次 `get_batch` 消费后，返回结构中也会附带两个「消费后剩余」字段（[message_queue.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/message_queue.py) L212 附近）：

| 字段 | 含义 |
|---|---|
| `remaining_queue_size` | 本次 `get_batch` 取走一批之后，队列中剩下的 item 数 |
| `remaining_pending_samples` | 本次 `get_batch` 取走一批之后，队列中剩下的 response 数 |

这两个字段目前只在 [fully_async_trainer.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/fully_async_trainer.py) L344 附近被读取并用于 `print` 日志，**未自动写入 logger**。如需要长期观察队列水位趋势（例如检查生产/消费是否均衡、是否长期接近 `max_queue_size` 导致 dropped 激增），可以在业务侧手工补一行 `self.logger.log({"fully_async/queue/size": remaining_queue, ...})`。

#### 5.6.3 诊断用法

- **队列长期打满**：`queue_size ≈ max_queue_size` 且 `dropped_samples` 持续增长 → Rollout 快于 Train，考虑调大 `max_queue_size`、收紧 `staleness_threshold` 以降低堆积，或降低 rollout 并发。
- **队列长期空置**：`queue_size ≈ 0` 且 `fully_async/trainer/idle_ratio` 高 → Rollout 是瓶颈，扩容 Rollouter。
- **`pending_samples` 远大于 `queue_size * rollout.n_default`**：说明存在 `has_at_least_positive` / adaptive rollout_n 等变长策略生效，推动单 item 携带更多 response。

---

## 6. `fully_async/staleness/*` — 参数版本 Staleness

由 `compute_staleness_metrics`（[detach_utils.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/detach_utils.py)）计算，反映 off-policy 程度。

- Response 级 staleness = `current_param_version - min_global_steps`（该 response 开始生成时的参数版本）。
- Prompt 级 staleness = 同一 prompt 下多条 response staleness 的 `max`（默认）或 `mean` 聚合（由 `rollout_ns` 切片）。

| 指标 | 含义 |
|---|---|
| `fully_async/staleness/response_staleness_mean` / `_max` / `_min` | Batch 内所有 response 级 staleness 的均值 / 最大 / 最小 |
| `fully_async/staleness/response_rollout_span_mean` / `_max` | Rollout 过程中跨越的参数版本数 `max_global_steps - min_global_steps` 的均值 / 最大 |
| `fully_async/staleness/prompt_staleness_mean` / `_max` / `_min` | Prompt 级（组内聚合后）staleness 统计 |

---

## 7. `fully_async/prompt_buffer/*` — Prompt Buffer 观测

由 `assemble_batch_from_rollout_samples` 在组 batch 时写入，只统计 `pass_rate >= 0`（已被观测过）的 prompt。

| 指标 | 含义 |
|---|---|
| `fully_async/prompt_buffer/pass_rate_mean` / `_std` / `_min` / `_max` | Prompt buffer 中被采样 prompt 的历史 pass_rate 统计 |
| `fully_async/prompt_buffer/sample_count_mean` / `_max` | 被采样 prompt 在 prompt buffer 中已累计的 sample 次数 |
| `fully_async/prompt_buffer/sampling_prob_mean` / `_std` / `_max` / `_min` | Priority Sampling 下每个 prompt 的采样概率（均匀采样时固定为 `1/|buffer|`） |
| `fully_async/prompt_buffer/rollout_n_mean` | 平均每个 prompt 的 rollout 次数（Adaptive rollout_n 时动态变化） |
| `fully_async/prompt_buffer/unique_prompt_uids` | 本 batch 中涉及的不同 prompt 个数（检测 rollout_n 膨胀） |

---

## 8. `fully_async/processing_time/*` 与 `fully_async/rollouter/*`

| 指标 | 含义 |
|---|---|
| `fully_async/processing_time/avg` / `max` / `min` / `tp50` / `tp95` / `tp99` | 单条 response `generate_sequences` 耗时分布 |
| `timing_s/agent_loop/tool_calls/max` / `min` / `mean` | 多轮工具调用耗时（多轮场景下） |
| `fully_async/rollouter/active_time` | Rollouter 实际处于生成状态的累计时间 |
| `fully_async/trainer/idle_ratio` | `timing_s/gen / timing_s/step`，trainer 等待 rollouter 的时间占比，越高说明 rollout 是瓶颈 |

---

## 9. `batch_analysis/*` — Batch-level 训练信号分析

由 [utils/batch_metrics.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/utils/batch_metrics.py) 中 `compute_batch_training_signal_metrics` 统一产出。

### 9.1 难度分箱 `batch_analysis/difficulty/*`

基于 prompt buffer 的历史 `pass_rate`（`p_i`）划分 5+1 个桶：

| Bucket | 判据 | 含义 |
|---|---|---|
| `extremely_hard` | `p_i == 0` | 从未答对 |
| `hard` | `0 < p_i <= 0.2` | 偶尔答对 |
| `medium` | `0.2 < p_i < 0.8` | **训练信号最密集区域** |
| `easy` | `0.8 <= p_i < 1.0` | 几乎总能答对 |
| `extremely_easy` | `p_i == 1.0` | 总是答对 |
| `unknown` | `p_i < 0` | 尚未观测 |

每桶都会产出 `{bucket}` 绝对数量与 `{bucket}_ratio` 比例两组指标。

### 9.2 Advantage 信号 `batch_analysis/advantage/*`

| 指标 | 含义 |
|---|---|
| `total_prompts` / `total_samples` | batch 中的 prompt / sample 总数 |
| `nonzero_adv_prompts` / `nonzero_adv_samples` | 存在非零 advantage 的 prompt / sample 数 |
| `nonzero_adv_prompt_ratio` / `nonzero_adv_sample_ratio` | 对应比例，**过低说明 batch 中大量 sample 提供不了梯度信号** |

### 9.3 Response 长度 & Overlong `batch_analysis/response_length/*`

使用 `sequence_score = token_level_scores.sum(-1)` 拆分 correct（`>0`）/ incorrect（`<=0`）两组：

| 指标 | 含义 |
|---|---|
| `correct_count` / `incorrect_count` / `correct_ratio` | 两组样本数及 correct 占比 |
| `{correct,incorrect}/mean` / `max` / `min` | 响应长度分组统计 |
| `{correct,incorrect}/overlong_count` | 达到 `max_response_length` 被截断的 sample 数 |
| `{correct,incorrect}/overlong_ratio` | **超长截断比例**（对 DAPO 等长链路训练尤其关键） |

---

## 10. 指标聚合规则（`MetricsAggregator`）

在多 trainer step 汇总到单次 report 时，`MetricsAggregator._get_aggregation_type` 会按如下优先级决定聚合方式（[detach_utils.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/detach_utils.py)）：

1. **显式规则**：`fully_async/count/*`、`training/global_step` → `last`（取最后一步）；`timing_s/agent_loop/tool_calls/{min,max,mean}` → 对应 min/max/avg。
2. **关键字匹配**（按名称推断）：
   - 含 `timing_s/` → `time_sum`
   - 含 `mean/avg/average` → `avg`
   - 含 `max/maximum` → `max`
   - 含 `min/minimum` → `min`
   - 含 `sum/total` → `sum`
   - 含 `weighted_avg` → 按 `sample_counts` 加权平均
3. **默认** → `avg`。

另有若干特殊后处理（`_special_metrics_aggergate`）：

- `global_seqlen/minmax_diff = global_seqlen/max - global_seqlen/min`
- `perf/throughput = perf/total_num_tokens / (perf/time_per_step * total_gpus)`
- `fully_async/trainer/idle_ratio = timing_s/gen / timing_s/step`

---

## 11. 诊断速查表

| 现象 | 关注指标 | 可能原因 / 动作 |
|---|---|---|
| Trainer 空转率高 | `fully_async/trainer/idle_ratio`、`fully_async/rollouter/active_time` | Rollout 是瓶颈，扩容 Rollouter 或降低 rollout_n |
| Staleness 增大 | `fully_async/staleness/response_staleness_max` | 队列堆积 / 参数同步频率偏低，考虑更激进的 purge |
| Purge 频繁触发 | `fully_async/purge/purged_samples` | Rollout 远快于 Train，考虑放宽 `staleness_threshold` 或减少 rollout 并发 |
| 队列被打满丢样 | `fully_async/count/dropped_samples`（= `MessageQueue.dropped_samples`） | `max_queue_size` 或 trainer 消费速率不足 |
| 训练信号稀疏 | `batch_analysis/advantage/nonzero_adv_sample_ratio` 偏低 | 开启 `rejection_sampling`、引入 Priority Sampling |
| 数据过易 | `fully_async/rejection_sampling/prompt/solve_all_ratio` 高 | 引入更难 prompt 或启用课程学习 |
| 数据过难 | `fully_async/rejection_sampling/prompt/solve_none_ratio` 高 | 降低难度、启用 SFT 预热 |
| 长度截断严重 | `batch_analysis/response_length/{correct,incorrect}/overlong_ratio` | 调大 `max_response_length` 或 DAPO 长度惩罚 |
| Pad 浪费多 | `fully_async/pad/pad_size / pre_pad_sequences` | 调整 `ppo_mini_batch_size` 或固定 rollout 数 |
| Raw vs Accepted acc 差距大 | `batch_analysis/raw_acc/mean` vs 训练 acc | 说明 rejection 过滤力度强，评估训练代表性 |

---

## 附：主要指标生成位置速查

| 模块 | 产出的主要 key 前缀 |
|---|---|
| [detach_utils.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/detach_utils.py) `assemble_batch_from_rollout_samples` | `fully_async/partial/*`, `fully_async/prompt_buffer/*`, `fully_async/processing_time/*` |
| [detach_utils.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/detach_utils.py) `compute_staleness_metrics` | `fully_async/staleness/*` |
| [fully_async_trainer.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/fully_async_trainer.py) batch 组装尾部 | `fully_async/pad/*`, `fully_async/rejection_sampling/*` |
| [fully_async_trainer.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/fully_async_trainer.py) `_fit_update_weights` 末尾 | `fully_async/purge/*`（仅在 `staleness_threshold>0` 且有命中时写入） |
| [message_queue.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/message_queue.py) `get_statistics` / `get_batch` 返回值 | 队列 size / pending / dropped（**未自动打点**，见 §5.6） |
| [fully_async_rollouter.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/fully_async_rollouter.py) Rollouter 主循环 | `fully_async/count/*`, `fully_async/rollouter/*` |
| [utils/batch_metrics.py](/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/utils/batch_metrics.py) | `batch_analysis/*`（含 `raw_acc` / `unweighted_acc`） |

