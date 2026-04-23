# Copyright 2025 Tencent Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AsyncResponseBuffer - Tracks in-flight & finished sequences per prompt_uid for
the asynchronous adaptive rollout scheme.

Responsibilities:
    - Track per-prompt_uid state: in-flight count, finished sequences with their rewards.
    - Provide stop_generation() to decide whether a prompt's rollout should stop.
    - Aggregate all finished sequences of a stopped prompt into a single bundle,
      ready to be wrapped into one RolloutSample for the trainer.

Thread-safety:
    This buffer itself is NOT thread / asyncio safe. It relies on the following
    assumptions for correctness in the fully-async rollouter:
      1. All callers run inside the SAME asyncio event loop (single-threaded).
      2. For a given prompt_uid, mutations (on_response_finished, on_job_submitted,
         stop_generation, mark_if_complete, pop_completed_prompt) are called
         sequentially within one ``_process_one_response`` coroutine — no two
         coroutines ever mutate the same PromptState concurrently.
      3. Different prompt_uids operate on independent PromptState objects; the
         shared ``_states`` dict is only mutated via atomic CPython dict ops
         (protected by the GIL).
    If any of these assumptions are violated (e.g. multi-threaded access, or
    external callers outside the event loop), an asyncio.Lock MUST be added
    around the mutation sequences.

The ``train_trigger`` decision (``fixed_prompt`` vs ``fixed_samples``) is
    NOT owned by this buffer -- it lives on the trainer side, which inspects
    the message queue's pending-sequence count / RolloutSample count directly.
    We keep the ``train_trigger`` argument purely as a config record that the
    rollouter can forward to the trainer; it has no effect on buffer behavior.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


logger = logging.getLogger(__name__)


# Stop rules for per-prompt_uid rollout
STOP_RULE_FIXED = "fixed_rollout"
STOP_RULE_PREFIXED = "prefixed_rollout"
STOP_RULE_AT_LEAST_POS = "has_at_least_positive"
STOP_RULE_AT_LEAST_POS_NEG = "has_at_least_positive_and_negative"
STOP_RULE_MAX = "max_rollout"

_VALID_STOP_RULES = {
    STOP_RULE_FIXED,
    STOP_RULE_PREFIXED,
    STOP_RULE_AT_LEAST_POS,
    STOP_RULE_AT_LEAST_POS_NEG,
    STOP_RULE_MAX,
}

# Train-trigger rules
TRAIN_TRIGGER_FIXED_PROMPT = "fixed_prompt"
TRAIN_TRIGGER_FIXED_SAMPLES = "fixed_samples"


@dataclass
class PromptState:
    """Per-prompt_uid tracking state."""

    prompt_uid: str
    sampling_prob: float = 0.0
    # Number of sequence generation jobs currently in flight (submitted but not yet finished)
    in_flight: int = 0
    # Finished sequences and their rewards (aligned by index)
    sequences: list[Any] = field(default_factory=list)  # list[DataProto of length 1]
    rewards: list[float] = field(default_factory=list)
    # Whether this prompt is marked as stopped (no more new jobs should be issued)
    stopped: bool = False
    # Cached per-prompt target N for STOP_RULE_FIXED. Resolved lazily (once)
    # on the first stop_generation() call and reused for the remainder of this
    # prompt's lifetime, so the target never drifts mid-generation and the
    # dynamic_fixed_n_provider is never called more than once per prompt.
    target_n: Optional[int] = None
    # Policy parameter version active at the moment this prompt was first
    # registered (i.e. its first sequence job was about to be submitted).
    # Used to compute per-prompt staleness =
    #     current_param_version - start_param_version
    # which drives the prompt-level staleness pause in the rollouter.
    start_param_version: int = 0

    @property
    def num_finished(self) -> int:
        return len(self.rewards)

    @property
    def num_positive(self) -> int:
        return sum(1 for r in self.rewards if r > 0)

    @property
    def num_negative(self) -> int:
        return sum(1 for r in self.rewards if r <= 0)

    @property
    def is_complete(self) -> bool:
        """Stopped and no sequence still in flight -> ready to emit."""
        return self.stopped and self.in_flight == 0


class AsyncResponseBuffer:
    """Per-prompt response buffer for async adaptive rollout.

    Args:
        stop_rule: Per-prompt stop rule, one of STOP_RULE_*.
        fixed_rollout_n: Static N used by STOP_RULE_FIXED when no dynamic provider
            is given (stop once finished >= N).
        max_rollout_n: Global cap on # sequences per prompt for all rules.
        train_trigger: "fixed_prompt" or "fixed_samples". Recorded here for
            introspection / logging only -- the actual trigger decision lives
            on the trainer side (via the message queue's pending-sequence /
            pending-RolloutSample counters).
        dynamic_fixed_n_provider: Optional callable ``prompt_uid -> int`` that
            returns a per-prompt N for STOP_RULE_FIXED (e.g. a bell-shaped
            function of pass_rate). When provided, overrides ``fixed_rollout_n``
            on a per-prompt basis. Returned values are clamped to [1, max_rollout_n].
            Called AT MOST ONCE per prompt (result cached in ``PromptState.target_n``).
    """

    def __init__(
        self,
        stop_rule: str = STOP_RULE_FIXED,
        fixed_rollout_n: int = 8,
        max_rollout_n: int = 16,
        train_trigger: str = TRAIN_TRIGGER_FIXED_PROMPT,
        dynamic_fixed_n_provider: Optional[Callable[[str], int]] = None,
    ):
        if stop_rule not in _VALID_STOP_RULES:
            raise ValueError(
                f"Invalid stop_rule={stop_rule}. Must be one of {sorted(_VALID_STOP_RULES)}."
            )
        if train_trigger not in (TRAIN_TRIGGER_FIXED_PROMPT, TRAIN_TRIGGER_FIXED_SAMPLES):
            raise ValueError(
                f"Invalid train_trigger={train_trigger}. "
                f"Must be '{TRAIN_TRIGGER_FIXED_PROMPT}' or '{TRAIN_TRIGGER_FIXED_SAMPLES}'."
            )
        if fixed_rollout_n < 1 or max_rollout_n < 1:
            raise ValueError(
                f"fixed_rollout_n ({fixed_rollout_n}) and max_rollout_n ({max_rollout_n}) must be >= 1."
            )
        if fixed_rollout_n > max_rollout_n:
            raise ValueError(
                f"fixed_rollout_n ({fixed_rollout_n}) must not exceed max_rollout_n ({max_rollout_n})."
            )

        self.stop_rule = stop_rule
        self.fixed_rollout_n = fixed_rollout_n
        self.max_rollout_n = max_rollout_n
        self.train_trigger = train_trigger
        self.dynamic_fixed_n_provider = dynamic_fixed_n_provider

        # prompt_uid -> PromptState
        self._states: dict[str, PromptState] = {}

        # Aggregated statistics
        self._completed_prompts_pending: list[str] = []  # prompt_uids waiting to be popped for training
        self._pending_samples: int = 0  # sum of samples inside _completed_prompts_pending

        logger.info(
            "[AsyncResponseBuffer] init: stop_rule=%s, fixed_rollout_n=%d, max_rollout_n=%d, "
            "train_trigger=%s, dynamic_fixed_n_provider=%s",
            stop_rule,
            fixed_rollout_n,
            max_rollout_n,
            train_trigger,
            "enabled" if dynamic_fixed_n_provider is not None else "disabled",
        )

    # ------------------------------------------------------------------
    # Life-cycle: register, submit, finish
    # ------------------------------------------------------------------

    def register_prompt(
        self,
        prompt_uid: str,
        sampling_prob: float = 0.0,
        start_param_version: int = 0,
    ) -> PromptState:
        """Ensure a state entry exists for ``prompt_uid`` and return it.

        This is typically called when the first sequence job of a new prompt is
        about to be submitted. Safe to call multiple times for the same uid.

        ``start_param_version`` records the rollouter's current policy parameter
        version at registration time; it is used downstream to compute
        prompt-level staleness. The value is set ONLY on first registration --
        subsequent calls with the same uid do not overwrite it, so the prompt's
        "birth version" is stable for its whole lifetime.
        """
        if prompt_uid not in self._states:
            self._states[prompt_uid] = PromptState(
                prompt_uid=prompt_uid,
                sampling_prob=sampling_prob,
                start_param_version=int(start_param_version),
            )
        return self._states[prompt_uid]

    def on_job_submitted(self, prompt_uid: str) -> None:
        """Mark that a new sequence job has been submitted for this prompt."""
        state = self._states.get(prompt_uid)
        if state is None:
            state = self.register_prompt(prompt_uid)
        state.in_flight += 1

    def on_response_finished(
        self,
        prompt_uid: str,
        sequence: Any,
        reward: float,
    ) -> None:
        """Record a finished sequence (single sample DataProto) and its reward."""
        state = self._states.get(prompt_uid)
        if state is None:
            # Shouldn't happen if register/submit is called; create a state just in case.
            state = self.register_prompt(prompt_uid)
            logger.warning(
                "[AsyncResponseBuffer] on_response_finished called for unregistered prompt_uid=%s",
                prompt_uid,
            )
        state.in_flight = max(0, state.in_flight - 1)
        state.sequences.append(sequence)
        state.rewards.append(float(reward))

    # ------------------------------------------------------------------
    # Stop rule
    # ------------------------------------------------------------------

    def _resolve_fixed_n(self, prompt_uid: str) -> int:
        """Resolve the fixed_rollout_n threshold for this prompt.

        Result is cached on ``PromptState.target_n`` so we only ever call
        ``dynamic_fixed_n_provider`` ONCE per prompt. This has two benefits:
          * performance -- avoids a dict lookup + float() + strategy dispatch
            on every on_response_finished callback;
          * semantic stability -- the per-prompt target N is locked at the
            moment we first need to check it (typically right after the first
            finished sequence), so late pass_rate updates on a *different*
            co-running prompt won't cause this prompt's target N to jitter
            mid-generation (which would break the GRPO group-size contract
            downstream when stop_rule=fixed_rollout).

        If a dynamic provider is installed (e.g. bell-shaped based on pass_rate),
        query it on first access; otherwise fall back to the static
        ``fixed_rollout_n``. Result is clamped to [1, max_rollout_n].
        """
        state = self._states.get(prompt_uid)
        if state is not None and state.target_n is not None:
            return state.target_n

        n = self.fixed_rollout_n
        if self.dynamic_fixed_n_provider is not None:
            try:
                n = int(self.dynamic_fixed_n_provider(prompt_uid))
            except Exception as e:
                logger.warning(
                    "[AsyncResponseBuffer] dynamic_fixed_n_provider raised for prompt_uid=%s: %s; "
                    "falling back to static fixed_rollout_n=%d",
                    prompt_uid,
                    e,
                    self.fixed_rollout_n,
                )
                n = self.fixed_rollout_n
        resolved = max(1, min(int(n), self.max_rollout_n))
        if state is not None:
            state.target_n = resolved
        return resolved

    def stop_generation(self, prompt_uid: str) -> bool:
        """Decide whether to stop issuing new sequence jobs for ``prompt_uid``.

        Returns True once the configured rule is satisfied OR the hard max is
        reached. Idempotent - once True, it remains True (via state.stopped).
        """
        state = self._states.get(prompt_uid)
        if state is None:
            return True  # unknown uid -> definitely stop

        if state.stopped:
            return True

        # Hard cap
        if state.num_finished >= self.max_rollout_n:
            state.stopped = True
            return True

        rule = self.stop_rule
        stop = False
        if rule == STOP_RULE_FIXED:
            threshold = self._resolve_fixed_n(prompt_uid)
            stop = state.num_finished >= threshold
        elif rule == STOP_RULE_PREFIXED:
            # Stop when finished >= ceil(1 / p_i), capped at max_rollout_n
            p = max(state.sampling_prob, 1e-8)
            threshold = min(self.max_rollout_n, int(math.ceil(1.0 / p)))
            stop = state.num_finished >= max(1, threshold)
        elif rule == STOP_RULE_AT_LEAST_POS:
            stop = state.num_positive >= 1
        elif rule == STOP_RULE_AT_LEAST_POS_NEG:
            stop = state.num_positive >= 1 and state.num_negative >= 1
        elif rule == STOP_RULE_MAX:
            stop = state.num_finished >= self.max_rollout_n

        if stop:
            state.stopped = True
        return stop

    # ------------------------------------------------------------------
    # Completion: pop prompts whose generation is fully done
    # ------------------------------------------------------------------

    def mark_if_complete(self, prompt_uid: str) -> bool:
        """If the prompt is stopped AND has no in-flight jobs, push it to the
        pending-completed queue. Returns True iff it was just pushed.
        """
        state = self._states.get(prompt_uid)
        if state is None:
            return False
        if not state.is_complete:
            return False
        # Guard against double push
        if prompt_uid in self._completed_prompts_pending:
            return False
        self._completed_prompts_pending.append(prompt_uid)
        self._pending_samples += state.num_finished
        return True

    def pop_completed_prompt(self, prompt_uid: Optional[str] = None) -> Optional[PromptState]:
        """Pop a completed prompt's full state (for emitting a RolloutSample).

        If ``prompt_uid`` is given, pop that specific one (if completed);
        otherwise pop the oldest pending completed prompt.
        """
        if prompt_uid is not None:
            if prompt_uid not in self._completed_prompts_pending:
                return None
            self._completed_prompts_pending.remove(prompt_uid)
        else:
            if not self._completed_prompts_pending:
                return None
            prompt_uid = self._completed_prompts_pending.pop(0)

        state = self._states.pop(prompt_uid, None)
        if state is not None:
            self._pending_samples = max(0, self._pending_samples - state.num_finished)
        return state

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def num_active_prompts(self) -> int:
        """Prompts that are registered but not yet popped (i.e. in working pool)."""
        return len(self._states)

    def num_unstopped_prompts(self) -> int:
        """Prompts that have not been marked as stopped yet."""
        return sum(1 for s in self._states.values() if not s.stopped)

    def num_pending_completed_prompts(self) -> int:
        return len(self._completed_prompts_pending)

    def num_pending_samples(self) -> int:
        return self._pending_samples

    def get_state(self, prompt_uid: str) -> Optional[PromptState]:
        return self._states.get(prompt_uid)

    def max_live_prompt_staleness(self, current_param_version: int) -> int:
        """Return max(current_param_version - start_param_version) over all
        live prompts (i.e. registered but not yet popped). Returns 0 when no
        prompts are alive, so a freshly-drained rollouter is never considered
        stale.

        "Live" here matches ``num_active_prompts()`` -- every entry in
        ``_states``, including stopped-but-not-emitted prompts. We include
        stopped prompts on purpose: as long as they have not yet been popped
        (their sequences are still held by the buffer and will be emitted to
        the trainer), their birth version still contributes to how stale the
        next emitted RolloutSample can be.
        """
        if not self._states:
            return 0
        return max(
            int(current_param_version) - int(s.start_param_version)
            for s in self._states.values()
        )

    def num_stale_prompts(self, current_param_version: int) -> int:
        """Return the number of live prompts whose ``start_param_version`` is
        strictly less than ``current_param_version`` (i.e. staleness > 0).

        This is a prompt-level staleness counter: it counts how many prompts
        were born under an older parameter version and are still alive in the
        buffer (registered but not yet popped).
        """
        if not self._states:
            return 0
        cpv = int(current_param_version)
        return sum(
            1 for s in self._states.values()
            if cpv > int(s.start_param_version)
        )

    def count_stale_samples(self, current_param_version: int) -> int:
        """Return the total number of stale samples (in_flight + finished) held
        by this buffer, i.e. samples belonging to live prompts whose
        ``start_param_version < current_param_version``.

        This is a sample-level staleness counter intended to be called by the
        rollouter from ``reset_staleness()`` to recompute the running
        ``stale_samples`` accumulator after a parameter version bump. It does
        NOT include samples that have already been popped out of the buffer
        (those have already been emitted to the message queue and are governed
        by the queue's own purge mechanism instead).

        We deliberately keep this scoped to buffer-resident samples only:
        samples sitting in the downstream message queue may have arbitrary
        staleness in ``[0, staleness_threshold-1]`` (the ones with
        ``staleness >= staleness_threshold`` have already been purged by the
        trainer), so conflating them with buffer-side stale samples would make
        the counter's semantics fuzzy.
        """
        if not self._states:
            return 0
        cpv = int(current_param_version)
        total = 0
        for s in self._states.values():
            if cpv > int(s.start_param_version):
                total += s.in_flight + s.num_finished
        return total

    def stats(self) -> dict:
        total_in_flight = sum(s.in_flight for s in self._states.values())
        total_finished = sum(s.num_finished for s in self._states.values())
        return {
            "response_buffer/active_prompts": self.num_active_prompts(),
            "response_buffer/unstopped_prompts": self.num_unstopped_prompts(),
            "response_buffer/pending_completed_prompts": self.num_pending_completed_prompts(),
"response_buffer/pending_samples": self.num_pending_samples(),
            "response_buffer/in_flight_samples": total_in_flight,
            "response_buffer/live_finished_samples": total_finished,
        }
