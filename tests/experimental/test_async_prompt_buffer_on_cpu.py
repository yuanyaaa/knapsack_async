from copy import deepcopy

from verl.experimental.fully_async_policy.utils.async_prompt_buffer import (
    AsyncPromptBuffer,
    resolve_prompt_sample_strategy,
)


class _ListDataset:
    def __init__(self, size: int):
        self.rows = [{"extra_info": {"prompt_uid": str(i)}} for i in range(size)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return deepcopy(self.rows[index])


def _make_buffer(size: int, seed: int = 123, sample_strategy=None) -> AsyncPromptBuffer:
    return AsyncPromptBuffer(
        dataset=_ListDataset(size),
        collate_fn=lambda rows: rows[0],
        seed=seed,
        sample_strategy=sample_strategy,
    )


def _draw_prompt_indices(buffer: AsyncPromptBuffer, count: int) -> list[int]:
    return [int(buffer.get_single_sample()[1]) for _ in range(count)]


def test_shuffle_sequential_is_default_and_covers_each_epoch_once():
    size = 16
    buffer = _make_buffer(size=size, seed=7)

    first_epoch = _draw_prompt_indices(buffer, size)
    second_epoch = _draw_prompt_indices(buffer, size)

    assert buffer._sample_mode == "shuffle_sequential"
    assert sorted(first_epoch) == list(range(size))
    assert sorted(second_epoch) == list(range(size))
    assert first_epoch != list(range(size))
    assert second_epoch != first_epoch


def test_shuffle_sequential_resume_restores_order_cursor_and_rng():
    size = 11
    original = _make_buffer(size=size, seed=19)
    _draw_prompt_indices(original, size + 4)
    state = original.state_dict()

    expected = _draw_prompt_indices(original, 2 * size)

    resumed = _make_buffer(size=size, seed=999)
    resumed.load_state_dict(state)
    actual = _draw_prompt_indices(resumed, 2 * size)

    assert actual == expected


def test_explicit_sequential_preserves_dataset_order():
    buffer = _make_buffer(size=5, seed=7, sample_strategy="sequential")

    assert _draw_prompt_indices(buffer, 8) == [0, 1, 2, 3, 4, 0, 1, 2]


def test_resolve_prompt_sample_strategy_defaults_to_shuffle_sequential():
    assert resolve_prompt_sample_strategy({}) == "shuffle_sequential"
    assert resolve_prompt_sample_strategy({"sample_strategy": None}) == "shuffle_sequential"
    assert resolve_prompt_sample_strategy({"sample_strategy": "null"}) == "shuffle_sequential"
    assert resolve_prompt_sample_strategy({"sample_strategy": "shuffle_sequential"}) == "shuffle_sequential"
    assert resolve_prompt_sample_strategy({"sample_strategy": "sequential"}) == "sequential"
