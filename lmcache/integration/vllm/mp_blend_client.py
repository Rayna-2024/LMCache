# SPDX-License-Identifier: Apache-2.0
"""Small MP CacheBlend client for the vLLM Ascend proof of concept.

The selective connector uses vLLM's standard external-token allocation
and completes its coverage before
the ordinary tail forward. The MP server owns matching, prefetch and scatter.
"""

# Standard
from dataclasses import dataclass, replace
from typing import Any
import json
import time

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.forward_context import get_forward_context
from vllm.v1.request import RequestStatus
import torch

# First Party
from lmcache import torch_dev
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.lmcache_mp_metadata import (
    BlendScatterTask,
    LMCacheMPConnectorMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    ExtraConfigDefault,
    LMCacheMPSchedulerAdapter,
)
from lmcache.utils import init_logger
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    CBUnifiedLookupResult,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.token_codec import num_packed_tokens
from lmcache.v1.platform import resolve_kv_wrapper_factory
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend

logger = init_logger(__name__)


class MPBlendSchedulerAdapter(LMCacheMPSchedulerAdapter):
    """Use the server's unified lookup in place of the ordinary prefix RPC."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if len(self.req_clients) != 1:
            raise ValueError("The MP CacheBlend PoC requires one MP server")
        self._blend_client = next(iter(self.req_clients.values()))
        cfg = kwargs.get("extra_config") or {}
        self._blend_timeout = float(
            cfg.get("lmcache.mp.mq_timeout", ExtraConfigDefault.mq_timeout.default)
        )
        self._blend_pending: dict[
            str,
            tuple[
                IPCCacheServerKey,
                MessagingFuture[CBUnifiedLookupResult | None],
                float,
            ],
        ] = {}
        self._blend_resolved: dict[
            str, tuple[IPCCacheServerKey, CBUnifiedLookupResult]
        ] = {}

    def maybe_submit_lookup_request(
        self,
        request_id: str,
        packed_token_ids: bytes,
        cache_salt: str = "",
        request_configs: dict[str, Any] | None = None,
        reserve_last_token: bool = False,
        start: int = 0,
    ) -> bool:
        self._ensure_heartbeat_started()
        if not self.is_healthy or request_id in self._blend_pending:
            return False
        if request_id in self._blend_resolved:
            return False
        end = (
            (num_packed_tokens(packed_token_ids) - int(reserve_last_token))
            // self.lmcache_tokens_per_chunk
            * self.lmcache_tokens_per_chunk
        )
        key = IPCCacheServerKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=None,
            token_bytes=packed_token_ids,
            start=start,
            end=max(0, end),
            request_id=request_id,
            cache_salt=cache_salt,
            request_configs=request_configs,
        )
        future = self._blend_client.cb_unified_lookup(key, self.tp_size)
        self._blend_pending[request_id] = (
            key, future, time.monotonic() + self._blend_timeout
        )
        return True

    def check_lookup_result(self, request_id: str) -> int | None:
        if not self.is_healthy:
            self._blend_pending.pop(request_id, None)
            return 0
        resolved = self._blend_resolved.get(request_id)
        if resolved is not None:
            return resolved[1].prefix_coverage_tokens
        pending = self._blend_pending.get(request_id)
        if pending is None:
            return 0
        key, future, deadline = pending
        if time.monotonic() >= deadline:
            logger.warning("CacheBlend lookup timed out for %s", request_id)
            self._blend_pending.pop(request_id)
            return 0
        if not future.query():
            return None
        try:
            result = future.result()
        except Exception:
            logger.exception("CacheBlend lookup failed for %s", request_id)
            self._blend_pending.pop(request_id)
            return 0
        if result is None:
            # The server has submitted prefetch; poll the same request again.
            self._blend_pending[request_id] = (
                key,
                self._blend_client.cb_unified_lookup(key, self.tp_size),
                deadline,
            )
            return None
        self._blend_pending.pop(request_id)
        self._blend_resolved[request_id] = (key, result)
        return result.prefix_coverage_tokens

    def get_blend_lookup(
        self, request_id: str
    ) -> tuple[IPCCacheServerKey, CBUnifiedLookupResult] | None:
        """Return the finished lookup, including non-prefix ranges."""
        return self._blend_resolved.get(request_id)

    def cleanup_lookup_result(self, request_id: str) -> None:
        self._blend_pending.pop(request_id, None)
        self._blend_resolved.pop(request_id, None)

    def discard_blend_lookup(self, request_id: str) -> None:
        """Forget one request's completed lookup after its session ends."""
        self._blend_pending.pop(request_id, None)
        self._blend_resolved.pop(request_id, None)


@dataclass
class BlendCoverageTask(BlendScatterTask):
    """Allocated coverage, separate from actual prefix/non-prefix cache hits.

    computed_prefix_tokens was ready before this schedule. prefix_hit_tokens
    becomes ready only after prefix retrieval succeeds. coverage_end includes
    holes the selective executor must compute before the tail can run.
    """

    computed_prefix_tokens: int
    prefix_hit_tokens: int
    coverage_end: int


class LMCacheMPSelectiveConnector(LMCacheMPConnector):
    """Use the in-process load-spec lifecycle with the standard vLLM scheduler.

    Subclasses implement complete_blend for their model/device. No scheduler
    output is rewritten: external coverage is promised before allocation, then
    fulfilled synchronously in start_load_kv, including known-load-failure
    recomputation. Unknown transport outcomes abort the forward.
    """

    def __init__(
        self, vllm_config: Any, role: KVConnectorRole, kv_cache_config: Any = None
    ) -> None:
        if vllm_config.scheduler_config.async_scheduling:
            raise ValueError("MP CacheBlend PoC requires async_scheduling=False")
        super().__init__(vllm_config, role, kv_cache_config)
        self._blend_rpc_timeout = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache.mp.mq_timeout", ExtraConfigDefault.mq_timeout.default
        )
        if role == KVConnectorRole.SCHEDULER:
            self._coverage_plans: dict[str, BlendCoverageTask] = {}
        else:
            self._blend_rope_registered = False
            self._blend_rope_wrapper = None
            self._blend_scatter_succeeded: set[str] = set()
            self._blend_scatter_failed: set[str] = set()

    def create_lmcache_mp_scheduler_adapter(
        self, **adapter_kwargs: Any
    ) -> MPBlendSchedulerAdapter:
        return MPBlendSchedulerAdapter(**adapter_kwargs)

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Report coverage beyond the ready prefix, reserving the final token.

        Ordinary lookup accounting continues to describe real prefix hits.
        Repeated calls before allocation replace the pending load spec; they
        neither advance the request nor publish a scatter task.
        """
        self._coverage_plans.pop(request.request_id, None)
        count, asynchronous = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        if count is None:
            return count, asynchronous
        lookup = self.scheduler_adapter.get_blend_lookup(request.request_id)
        if lookup is None or self._skip_request_load(request):
            return count, asynchronous
        key, result = lookup
        prefix = result.prefix_coverage_tokens
        end = max(num_computed_tokens, min(prefix, request.num_tokens - 1))
        matches = []
        for match in sorted(result.non_prefix_segments, key=lambda m: m.cur_st):
            if match.cur_st >= end and match.cur_ed <= request.num_tokens - 1:
                matches.append(match)
                end = match.cur_ed
        if end <= num_computed_tokens:
            return 0, False
        self._coverage_plans[request.request_id] = BlendCoverageTask(
            key, matches, [], num_computed_tokens, prefix, end
        )
        logger.info(
            "MP_BLEND_EVENT %s",
            json.dumps({
                "event": "lookup",
                "request_id": request.request_id,
                "prefix_coverage_tokens": prefix,
                "planned_segments": [[m.cur_st, m.cur_ed] for m in matches],
                "ready_prefix_tokens": num_computed_tokens,
                "coverage_end": end,
            }),
        )
        return end - num_computed_tokens, False

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        """Bind the load spec to allocated blocks, retaining true hit accounting."""
        task = self._coverage_plans.get(request.request_id)
        if task is not None:
            if num_external_tokens != task.coverage_end - task.computed_prefix_tokens:
                raise ValueError(
                    "allocated external tokens disagree with blend coverage"
                )
            task.block_ids = [list(ids) for ids in blocks.get_block_ids()]
        # The base tracker controls actual prefix RETRIEVE ranges, lookup
        # locks and STORE watermarks.
        super().update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: Any) -> LMCacheMPConnectorMetadata:
        """Attach allocated specs to tail forwards without changing the schedule."""
        metadata = super().build_connector_meta(scheduler_output)
        for request_id in scheduler_output.num_scheduled_tokens:
            task = self._coverage_plans.pop(request_id, None)
            if task is not None:
                metadata.blend_scatter_tasks.append(task)
        return metadata

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        """Wait for actual loads and finish coverage before ordinary forward.

        Prefix loads use device futures directly, so synchronous failures do
        not enter the ordinary adapter's asynchronous invalid-block recovery.
        A false result is passed to complete_blend for TP-wide recomputation;
        transport exceptions propagate because late writes cannot be ruled out.
        """
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, LMCacheMPConnectorMetadata):
            raise TypeError("expected MP connector metadata")
        if not metadata.blend_scatter_tasks:
            return
        device = torch.device("npu", torch_dev.current_device())
        stream = torch_dev.current_stream(device)
        events = get_event_ipc_backend(device)
        retrieves = {
            entry.request_id: entry
            for entry in metadata.requests if entry.direction == "RETRIEVE"
        }
        for task in metadata.blend_scatter_tasks:
            if not isinstance(task, BlendCoverageTask):
                raise TypeError("selective connector requires a coverage task")
            req_id = task.key.request_id
            prefix_ok = True
            if task.prefix_hit_tokens > task.computed_prefix_tokens:
                entry = retrieves[req_id]
                op = entry.op
                before = events.create_event(device)
                events.record_event(before, stream)
                key = replace(
                    task.key, worker_id=self.worker_adapter.worker_id,
                    start=op.start, end=op.end,
                )
                prefix_ok = self.worker_adapter.req_client.retrieve(
                    key, self.worker_adapter.instance_id, [op.flat_block_ids],
                    events.export_event(before, device),
                    max(op.skip_first_n_tokens, task.computed_prefix_tokens - op.start),
                ).to_device_future(device=device, event_backend=events).wait_on_stream(
                    stream, timeout=self._blend_rpc_timeout
                )
            self._scatter_blend_tasks([task])
            scatter_ok = not task.matches or req_id in self._blend_scatter_succeeded
            self.complete_blend(task, prefix_ok and scatter_ok, forward_context)
            self._blend_scatter_succeeded.discard(req_id)
            self._blend_scatter_failed.discard(req_id)

    def complete_blend(
        self, task: BlendCoverageTask, loaded: bool, forward_context: Any
    ) -> None:
        """Finish coverage, including TP-consistent full recompute on load failure.

        Implementations must order all KV writes before the tail forward and
        raise on execution failure; returning promises valid KV through
        task.coverage_end. loaded describes this worker's terminal load result.
        """
        raise NotImplementedError

    def request_finished(
        self, request: Any, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        """Discard unconsumed coverage and let the base connector end the session."""
        self._coverage_plans.pop(request.request_id, None)
        self.scheduler_adapter.discard_blend_lookup(request.request_id)
        return super().request_finished(request, block_ids)

    def unregister_kv_caches(self) -> None:
        if self._blend_rope_registered:
            self.worker_adapter.req_client.cb_unregister_rope(
                self.worker_adapter.instance_id
            ).result(timeout=self._blend_rpc_timeout)
            self._blend_rope_registered = False
            self._blend_rope_wrapper = None
        super().unregister_kv_caches()

    def restore_kv_caches(self) -> None:
        super().restore_kv_caches()
        self._blend_rope_registered = False
        self._blend_rope_wrapper = None

    def _skip_request_load(self, request: Any) -> bool:
        # A preemption can recycle the same physical blocks. The server's
        # applied-range set cannot distinguish that from an intact scatter.
        return (
            request.status == RequestStatus.PREEMPTED
            or super()._skip_request_load(request)
        )

    def _scatter_blend_tasks(self, tasks: list[BlendScatterTask]) -> None:
        tasks = [task for task in tasks if task.matches]
        if not tasks:
            return
        if not self._blend_rope_registered:
            self._register_blend_rope()
        device = torch.device("npu", torch_dev.current_device())
        stream = torch_dev.current_stream(device)
        events = get_event_ipc_backend(device)
        for task in tasks:
            before = events.create_event(device)
            events.record_event(before, stream)
            key = replace(task.key, worker_id=self.worker_adapter.worker_id)
            future = self.worker_adapter.req_client.cb_retrieve_pre_computed(
                key,
                task.matches,
                task.block_ids,
                self.worker_adapter.instance_id,
                events.export_event(before, device),
            )
            # A transport failure has an unknown scatter outcome: abort the
            # forward instead of racing a late scatter with local recompute.
            done_handle, scatter_ran = future.result(timeout=self._blend_rpc_timeout)
            logger.info(
                "MP_BLEND_EVENT %s",
                json.dumps(
                    {
                        "event": "scatter_worker",
                        "request_id": key.request_id,
                        "worker_id": self.worker_adapter.worker_id,
                        "scatter_ran": scatter_ran,
                        "matches": len(task.matches),
                        "block_table_lengths": [len(ids) for ids in task.block_ids],
                    }
                ),
            )
            # False may also carry an event draining a failed partial scatter.
            # Recompute must wait for it just as a successful consumer does.
            if done_handle:
                events.wait_event(events.import_event(done_handle, device), stream)
            if scatter_ran:
                self._blend_scatter_succeeded.add(key.request_id)
            else:
                self._blend_scatter_failed.add(key.request_id)

    def _register_blend_rope(self) -> None:
        model = get_forward_context().model_instance
        rope = next(
            (
                module
                for module in model.modules()
                if hasattr(module, "cos_sin_cache")
                and hasattr(module, "is_neox_style")
            ),
            None,
        )
        if rope is None:
            raise RuntimeError("Qwen3 RoPE cache is unavailable for CacheBlend")
        cache = rope.cos_sin_cache
        wrapper = resolve_kv_wrapper_factory(cache.device.type)(cache)
        self.worker_adapter.req_client.cb_register_rope(
            self.worker_adapter.instance_id,
            [wrapper],
            int(rope.head_size),
            bool(rope.is_neox_style),
            [],
            [],
            [],
        ).result(timeout=self._blend_rpc_timeout)
        self._blend_rope_wrapper = wrapper
        self._blend_rope_registered = True
