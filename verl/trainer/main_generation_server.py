# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Generate responses given a dataset of prompts
"""

import os
import time

import aiohttp
import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

import asyncio
from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf
from openai.types.chat import ChatCompletion

from verl.utils.hdfs_io import makedirs
from verl.workers.rollout.replica import get_rollout_replica_class


async def start_server(config):
    tp_size = config.actor_rollout_ref.rollout.tensor_model_parallel_size
    num_replicas = (config.trainer.n_gpus_per_node * config.trainer.nnodes) // tp_size
    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model
    # create standalone rollout server
    rollout_server_class = get_rollout_replica_class(config.actor_rollout_ref.rollout.name)
    rollout_servers = [
        rollout_server_class(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=config.trainer.n_gpus_per_node,
        )
        for replica_rank in range(num_replicas)
    ]
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])

    server_handles = [server._server_handle for server in rollout_servers]
    server_addresses = [server._server_address for server in rollout_servers]
    assert len(server_handles) == num_replicas
    assert len(server_addresses) == num_replicas

    return server_handles, server_addresses


async def submit_request(server_address, **chat_complete_request):
    try:
        extra_headers = chat_complete_request.pop("extra_headers", {})
        timeout = aiohttp.ClientTimeout(total=None)
        session = aiohttp.ClientSession(timeout=timeout)
        async with session.post(
            url=f"http://{server_address}/v1/chat/completions",
            headers={"Authorization": "Bearer token-abc123", **extra_headers},
            json=chat_complete_request,
        ) as resp:
            data = await resp.json()
            return ChatCompletion(**data)
    finally:
        await session.close()


async def generate_per_replica(server_address, model_path: str, n_samples: int, sampling_params: dict, chat_lst: list):
    # here we should sample n_samples for each chat_lst.
    # we use aiohttp to avoid hang in AsyncOpenAI when the number of requests is large.

    # client = AsyncOpenAI(
    #     api_key="123-abc",
    #     base_url=f"http://{server_address}/v1",
    # )

    chat_complete_request = [
        {
            "model": model_path,
            "messages": messages,
            **sampling_params,
        }
        for messages in chat_lst
        for _ in range(n_samples)
    ]

    tasks = [submit_request(server_address, **req) for req in chat_complete_request]
    results = await asyncio.gather(*tasks)
    return results


async def generate(
    server_addresses: list, model_path: str, n_samples: int, sampling_params: dict, chat_numpy: np.ndarray
):
    num_replicas = len(server_addresses)
    chat_sub_array = np.array_split(chat_numpy, num_replicas)
    chat_sub_array = [chat.tolist() for chat in chat_sub_array]
    assert len(server_addresses) == len(chat_sub_array)
    results = await asyncio.gather(
        *[
            generate_per_replica(server_addresses[i], model_path, n_samples, sampling_params, chat_sub_array[i])
            for i in range(num_replicas)
        ]
    )
    return results


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_USE_V1": "1"}})

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    n_samples = config.actor_rollout_ref.rollout.n

    if config.actor_rollout_ref.rollout.temperature == 0.0:
        assert n_samples == 1, "When temperature=0, n_samples must be 1."
    assert n_samples >= 1, "n_samples should always >= 1"

    sampling_params = {
        "temperature": config.actor_rollout_ref.rollout.temperature,
        "top_p": config.actor_rollout_ref.rollout.top_p,
        # "top_k": config.actor_rollout_ref.rollout.top_k,
        "max_tokens": config.actor_rollout_ref.rollout.response_length,
    }

    from omegaconf import ListConfig

    train_files = config.data.train_files
    if not isinstance(train_files, list | ListConfig):
        train_files = [train_files]

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)

    datasets = []
    for train_file in train_files:
        dataset = pd.read_parquet(train_file)
        datasets.append(dataset)

    # concat dataset
    dataset = pd.concat(datasets, axis=0, ignore_index=True)
    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [chat.tolist() for chat in chat_lst]
    chat_numpy = np.array(chat_lst)

    # Determine val_batch_size (number of prompts per batch)
    val_batch_size = getattr(config.data, "val_batch_size", None)
    total_prompts = len(chat_lst)
    if val_batch_size is None or val_batch_size <= 0:
        val_batch_size = total_prompts  # Process all at once
    val_batch_size = min(val_batch_size, total_prompts)
    num_batches = (total_prompts + val_batch_size - 1) // val_batch_size

    print(f"\n{'=' * 60}")
    print(f"  Generation Config")
    print(f"{'=' * 60}")
    print(f"  Total prompts:    {total_prompts}")
    print(f"  N samples/prompt: {n_samples}")
    print(f"  Val batch size:   {val_batch_size}")
    print(f"  Num batches:      {num_batches}")
    print(f"  Sampling params:  {sampling_params}")
    print(f"{'=' * 60}\n")

    # start native server
    t_server_start = time.time()
    server_handles, server_addresses = asyncio.run(start_server(config))
    t_server_elapsed = time.time() - t_server_start
    print(f"[Server] Started {len(server_addresses)} replica(s) in {t_server_elapsed:.1f}s")

    # run generate in batches
    import itertools

    all_results = []
    t_gen_start = time.time()

    for batch_idx in range(num_batches):
        batch_start = batch_idx * val_batch_size
        batch_end = min(batch_start + val_batch_size, total_prompts)
        batch_size = batch_end - batch_start
        batch_chat_numpy = chat_numpy[batch_start:batch_end]

        t_batch_start = time.time()
        print(f"[Batch {batch_idx + 1}/{num_batches}] Generating {batch_size} prompts x {n_samples} samples ...")

        gen_results = asyncio.run(
            generate(server_addresses, config.actor_rollout_ref.model.path, n_samples, sampling_params, batch_chat_numpy)
        )

        # flatten results from all replicas
        batch_results = list(itertools.chain.from_iterable(gen_results))
        all_results.extend(batch_results)

        t_batch_elapsed = time.time() - t_batch_start
        t_total_so_far = time.time() - t_gen_start
        prompts_done = batch_end
        prompts_remaining = total_prompts - prompts_done
        # Estimate remaining time
        avg_time_per_prompt = t_total_so_far / prompts_done if prompts_done > 0 else 0
        eta = avg_time_per_prompt * prompts_remaining

        print(
            f"[Batch {batch_idx + 1}/{num_batches}] Done in {t_batch_elapsed:.1f}s | "
            f"Progress: {prompts_done}/{total_prompts} ({prompts_done / total_prompts * 100:.1f}%) | "
            f"Elapsed: {t_total_so_far:.1f}s | ETA: {eta:.1f}s"
        )

    t_gen_elapsed = time.time() - t_gen_start
    print(f"\n[Generation] All batches complete in {t_gen_elapsed:.1f}s "
          f"({total_prompts} prompts, {total_prompts * n_samples} total responses)")

    # extract content from results
    results = np.array([result.choices[0].message.content for result in all_results])
    results = np.reshape(results, (-1, n_samples))

    assert results.shape == (len(chat_lst), n_samples)

    results = results.tolist()

    # add to the data frame
    dataset["responses"] = results

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {config.data.output_path}")
    t_save_start = time.time()
    dataset.to_parquet(config.data.output_path)
    t_save_elapsed = time.time() - t_save_start

    # Print time summary
    t_total = time.time() - t_server_start
    print(f"\n{'=' * 60}")
    print(f"  Generation Complete")
    print(f"{'=' * 60}")
    print(f"  Server startup:  {t_server_elapsed:.1f}s")
    print(f"  Generation:      {t_gen_elapsed:.1f}s")
    print(f"  Save parquet:    {t_save_elapsed:.1f}s")
    print(f"  Total:           {t_total:.1f}s")
    print(f"  Output:          {config.data.output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
