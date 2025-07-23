import time
from typing import TYPE_CHECKING, Optional, Union

import torch
import numpy as np
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.mla_v1 import AscendMLAMetadata
from vllm_ascend.attention.attention_v1_torchair import AscendTorchairMetadata
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_ND,
    ACL_FORMAT_FRACTAL_NZ,
    ProfileExecuteDuration,
    vllm_version_is,
    maybe_converting_weight_acl_format,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

from .utils import PatchHelper


class NPUModelRunnerPatch(PatchHelper[NPUModelRunner]):
    _orig_load_model = NPUModelRunner.load_model

    def patch_forward(self):
        from vllm.distributed.parallel_state import _SP
        sp_size = _SP.world_size
        sp_rank = _SP.rank_in_group
        device_group = _SP.device_group
        model_forward = self.model.forward

        def ulysses_forward(*args, **kwargs):
            input_ids = kwargs["input_ids"]
            positions = kwargs["positions"]
            # ulysses parameters
            N = input_ids.shape[0]
            N_chunk_size = N // sp_size
            N_offset = N_chunk_size * sp_rank
            # chunk the input
            kwargs["input_ids"] = input_ids[N_offset:N_offset + N_chunk_size]
            kwargs["positions"] = positions[N_offset:N_offset + N_chunk_size]
            output = model_forward(*args, **kwargs)
            # all gather model output
            model_output = torch.empty((N, self.model.config.hidden_size),
                                       dtype=output.dtype,
                                       device=output.device)
            torch.distributed.all_gather_into_tensor(model_output,
                                                     output,
                                                     group=device_group)
            return model_output
        
        self.model.forward = ulysses_forward

    def load_model(self, *args, **kwargs):
        self._orig_load_model(*args, **kwargs)
        if self.parallel_config.ulysses_sequence_parallel_size > 1:
            self.patch_forward()

    def _process_reqs(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> tuple[Union[AscendMetadata, AscendMLAMetadata,
                     AscendTorchairMetadata], torch.Tensor, SpecDecodeMetadata,
               torch.Tensor, int, torch.Tensor, torch.Tensor, np.ndarray]:
        # Check input valid
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        if (self.use_aclgraph and total_num_scheduled_tokens
                <= self.aclgraph_batch_sizes[-1]):
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                total_num_scheduled_tokens)
        else:
            # Eager mode.
            num_input_tokens = total_num_scheduled_tokens

        modified_batch = self.attn_metadata_builder.reorder_batch(
            self.input_batch, scheduler_output)
        if modified_batch:
            self.input_batch.refresh_sampling_metadata()

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        if vllm_version_is("0.9.2"):
            self.input_batch.block_table.commit(num_reqs)
        else:
            self.input_batch.block_table.commit_block_table(num_reqs)

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = np.empty(num_reqs, dtype=np.int32)
        num_valid_tokens = np.empty(num_reqs, dtype=np.int32)
        max_num_scheduled_tokens = 0
        for i, req_id in enumerate(self.input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens[i] = num_tokens
            num_valid_tokens[i] = num_tokens - \
                len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        # Prepare positions
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)
        cu_num_tokens = np.cumsum(num_scheduled_tokens)
        cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens,
                                    num_scheduled_tokens)
        logits_indices = cu_num_tokens - 1
        logits_indices = torch.from_numpy(logits_indices).to(self.device,
                                                             non_blocking=True)
        arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets

        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)

        self.positions[total_num_scheduled_tokens:num_input_tokens].zero_()
        self.positions[:total_num_scheduled_tokens].copy_(
            self.positions_cpu[:total_num_scheduled_tokens], non_blocking=True)
        positions = self.positions[:num_input_tokens]
        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        seq_lens = self.seq_lens_cpu[:num_reqs]

        block_table_indices = (req_indices * self.max_num_blocks_per_req +
                               positions_np // self.block_size)

        block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
        block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
        block_offsets = positions_np % self.block_size
        np.add(block_numbers * self.block_size,
               block_offsets,
               out=self.slot_mapping_np[:total_num_scheduled_tokens])

        ascend_config = get_ascend_config()
        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if np.array_equal(self.seq_lens_np[:num_reqs], num_scheduled_tokens):
            attn_state = AscendAttentionState.PrefillNoCache
        # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
        elif np.all(num_scheduled_tokens == 1):
            attn_state = AscendAttentionState.DecodeOnly
        # Speculative decoding.
        elif np.all(num_valid_tokens == 1):
            if self.use_eagle:
                attn_state = AscendAttentionState.ChunkedPrefill
            else:
                attn_state = AscendAttentionState.SpecDecoding
        # splitfuse
        elif not ascend_config.ascend_scheduler_config.enabled or self.chunked_prefill_enabled:
            attn_state = AscendAttentionState.ChunkedPrefill
        else:
            attn_state = AscendAttentionState.PrefillCacheHit

        self.attn_mask = self._make_attention_mask(
            seq_lens=seq_lens,
            query_lens=num_scheduled_tokens,
            position=positions,
            attn_state=attn_state)
        self.attn_state = attn_state  # type: ignore

        extra_builder_kwargs = {}

        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens
        self.query_start_loc[:num_reqs + 1].copy_(
            self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                       non_blocking=True)

        # Fill unused with -1. Needed for reshape_and_cache
        self.seq_lens[num_reqs:].fill_(0)
        self.query_start_loc[num_reqs + 1:].fill_(-1)

        with_prefill = attn_state not in [
            AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding
        ]

        if self.dp_size > 1:
            max_num_tokens, with_prefill = self._get_forward_metadata_across_dp(
                total_num_scheduled_tokens, with_prefill)
            extra_builder_kwargs['max_num_tokens_across_dp'] = max_num_tokens
            extra_builder_kwargs['with_prefill_across_dp'] = with_prefill

        # Add graph_pad_size here
        if self.torchair_graph_enabled and not with_prefill:
            if self.dp_size > 1:
                padded_batch_size = self.select_torchair_padded_batch_size(
                    max_num_tokens)
            else:
                padded_batch_size = self.select_torchair_padded_batch_size(
                    total_num_scheduled_tokens)
            graph_pad_size = padded_batch_size - total_num_scheduled_tokens

            extra_builder_kwargs['graph_pad_size'] = graph_pad_size

        if self.vllm_config.model_config.use_mla:
            extra_builder_kwargs[
                "query_start_loc"] = self.query_start_loc[:num_reqs + 1]
            attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                **extra_builder_kwargs,
            )
            attn_metadata.num_input_tokens = num_input_tokens
        else:
            attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                **extra_builder_kwargs,
            )

        # Prepare input_ids
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])
        # Copy the tensors to the NPU.
        self.input_ids[:total_num_scheduled_tokens].copy_(
            self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        if self.is_multimodal_model:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:total_num_scheduled_tokens]
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:total_num_scheduled_tokens].copy_(
                inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the ACL graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]

        if self.torchair_graph_enabled and not with_prefill:
            input_ids = self.input_ids[:padded_batch_size]
            positions = self.positions[:padded_batch_size]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                self.intermediate_tensors[k][:num_input_tokens].copy_(
                    v[:num_input_tokens], non_blocking=True)
            intermediate_tensors = IntermediateTensors({
                k: v[:num_input_tokens]
                for k, v in self.intermediate_tensors.items()
            })

        # Run forward pass
        with set_forward_context(attn_metadata,
                                 self.vllm_config,
                                 num_tokens=num_input_tokens):
            with ProfileExecuteDuration().capture_async("forward"):
                model_kwargs = {}
                if self.torchair_graph_enabled:
                    model_kwargs["kv_caches"] = self.kv_caches
                    model_kwargs["attn_metadata"] = attn_metadata
                if self.torchair_graph_enabled and not with_prefill:
                    maybe_converting_weight_acl_format(self.model,
                                                       ACL_FORMAT_FRACTAL_NZ)

                    compiled_model = self._get_torchair_lazy_compiled_model(
                        padded_batch_size)
                    hidden_states = compiled_model(
                        input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                        **model_kwargs,
                    )
                else:
                    assert self.model is not None
                    maybe_converting_weight_acl_format(self.model,
                                                       ACL_FORMAT_FRACTAL_ND)

                    hidden_states = self.model(
                        input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                        **model_kwargs,
                    )

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            spec_decode_metadata = None
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens)
            logits_indices = spec_decode_metadata.logits_indices

        aux_hidden_states = None
        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = hidden_states

        return (attn_metadata, hidden_states, spec_decode_metadata, positions,
                total_num_scheduled_tokens, logits_indices, aux_hidden_states,
                num_scheduled_tokens)

    def capture_model(self) -> None:
        start_time = time.perf_counter()
        start_free_npu_memory = torch.npu.mem_get_info()[0]
        SP = self.parallel_config.sequence_parallel_size
        # TODO(NeverRaR): Calling graph_capture(device=self.device) in
        # torchair graph capture can cause some issues, so now we just
        # temporarily split the codepath for the two different graph patterns.
        if self.torchair_graph_enabled:
            torchair_graph_batch_sizes = self.torchair_graph_batch_sizes
            graph_num = len(torchair_graph_batch_sizes)
            logger.info(
                "Capturing torchair graph, this usually takes %.1f~%.1f mins.",
                0.5 * graph_num, 1.5 * graph_num)
            # Trigger torchair graph capture for specific shapes.
            # Capture the large shapes first so that the smaller shapes
            # can reuse the memory pool allocated for the large shapes.
            for idx, num_tokens in enumerate(
                    reversed(torchair_graph_batch_sizes)):
                for _ in range(self.vllm_config.compilation_config.
                               cudagraph_num_of_warmups):
                    # NOTE: when in torchair graph and not with_prefill,
                    # we don't need to set `skip_attn=False`
                    self._dummy_run(num_tokens * SP, is_torchair_compile=True)
                self._dummy_run(num_tokens * SP, is_torchair_compile=True)
                logger.info("Batchsize %d is compiled successfully: %d/%d.",
                            num_tokens, idx + 1, graph_num)
        elif self.use_aclgraph:
            # Trigger ACL graph capture for specific shapes.
            # Capture the large shapes first so that the smaller shapes
            # can reuse the memory pool allocated for the large shapes.
            # TODO(zzzzwwjj): Check dummy_run with ACL Graph and full graph mode
            from vllm_ascend.worker.model_runner_v1 import graph_capture
            with graph_capture(device=self.device):
                skip_attn = not self.vllm_config.compilation_config.full_cuda_graph
                # TODO: Make sure passing attn_state to _dummy_run in the future
                for num_tokens in reversed(self.aclgraph_batch_sizes):
                    for _ in range(self.vllm_config.compilation_config.
                                   cudagraph_num_of_warmups):
                        self._dummy_run(num_tokens * SP, skip_attn=skip_attn)
                    self._dummy_run(num_tokens * SP, skip_attn=skip_attn)
        else:
            logger.info("Skipping NPU graph capture for eager mode.")
            return
        end_time = time.perf_counter()
        end_free_npu_memory = torch.npu.mem_get_info()[0]
        elapsed_time = end_time - start_time
        npu_graph_size = start_free_npu_memory - end_free_npu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, npu_graph_size / (1 << 30))
