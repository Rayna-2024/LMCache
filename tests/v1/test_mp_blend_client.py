# SPDX-License-Identifier: Apache-2.0
"""CPU contract checks for the MP CacheBlend engine client."""

# Standard
from types import SimpleNamespace
import threading

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPConnectorMetadata,
)
from lmcache.integration.vllm.mp_blend_client import (
    MPBlendSchedulerAdapter,
)
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    CBUnifiedLookupResult,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.token_codec import pack_token_ids


def _match(start: int, end: int) -> CBMatchResult:
    return CBMatchResult(0, end - start, start, end, b"hit")


def test_unified_lookup_polls_until_sparse_result() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls: list[IPCCacheServerKey] = []

        def cb_unified_lookup(self, key: IPCCacheServerKey, tp_size: int):
            assert tp_size == 1
            self.calls.append(key)
            future = MessagingFuture()
            result = (
                None
                if len(self.calls) == 1
                else CBUnifiedLookupResult(4, [_match(8, 12)])
            )
            future.set_result(result)
            return future

    client = Client()
    adapter = object.__new__(MPBlendSchedulerAdapter)
    adapter.model_name = "qwen3"
    adapter.parallel_strategy = SimpleNamespace(kv_world_size=1, kv_tp_size=1)
    adapter.lmcache_tokens_per_chunk = 4
    adapter._health_events = {"server": threading.Event()}
    adapter._health_events["server"].set()
    adapter._blend_client = client
    adapter._blend_pending = {}
    adapter._blend_resolved = {}
    adapter._blend_timeout = 10
    adapter._ensure_heartbeat_started = lambda: None

    assert adapter.maybe_submit_lookup_request("r", pack_token_ids(range(13)))
    assert adapter.check_lookup_result("r") is None
    assert adapter.check_lookup_result("r") == 4
    key, result = adapter.get_blend_lookup("r")
    assert key.end == 12
    assert result.non_prefix_segments[0].cur_st == 8
    assert len(client.calls) == 2


@pytest.mark.parametrize("load_result", [True, False, "timeout"])
@pytest.mark.parametrize("sparse", [False, True])
def test_selective_load_completion_orders_tail_or_aborts(
    monkeypatch, load_result, sparse
):
    """A prefix-only hit still needs a wait; unknown writes cannot reach the tail."""
    from lmcache.integration.vllm import mp_blend_client as module
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata
    from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp
    from lmcache.v1.mp_observability.errors import LMCacheTimeoutError

    calls = []
    events = SimpleNamespace(
        create_event=lambda device: object(),
        record_event=lambda event, stream: None,
        export_event=lambda event, device: b"before",
        import_event=lambda handle, device: calls.append("import") or object(),
        wait_event=lambda event, stream: calls.append("wait"),
    )
    future = MessagingFuture()
    if load_result == "timeout":
        future.set_exception(LMCacheTimeoutError("unknown remote outcome"))
    else:
        future.set_result((b"done", load_result))
    client = SimpleNamespace(
        retrieve=lambda *args: future, cb_retrieve_pre_computed=lambda *args: future
    )
    connector = object.__new__(module.LMCacheMPSelectiveConnector)
    connector.worker_adapter = SimpleNamespace(
        req_client=client, worker_id=0, instance_id=1
    )
    connector._blend_rpc_timeout = 1
    connector._blend_rope_registered = True
    connector._blend_scatter_succeeded = set()
    connector._blend_scatter_failed = set()
    connector.complete_blend = (
        lambda task, loaded, context: calls.append(("complete", loaded))
    )
    monkeypatch.setattr(
        module, "torch_dev", SimpleNamespace(
            current_device=lambda: 0, current_stream=lambda device: object()
        )
    )
    monkeypatch.setattr(module, "get_event_ipc_backend", lambda device: events)
    # No device tensor, stream or context is created by this test.
    monkeypatch.setattr(module.torch, "device", lambda *args: "cpu")
    key = IPCCacheServerKey.from_token_ids("qwen3", 1, None, range(9), request_id="r")
    meta = LMCacheMPConnectorMetadata()
    meta.blend_scatter_tasks = [module.BlendCoverageTask(
        key, [_match(4, 8)] if sparse else [], [[1, 2, 3]], 0,
        0 if sparse else 8, 8,
    )]
    meta.requests = [LMCacheMPRequestMetadata(
        request_id="r", direction="RETRIEVE",
        op=LoadStoreOp(token_bytes=key.token_bytes, block_ids=[[1, 2]], start=0, end=8),
    )]
    if sparse:
        meta.requests = []
    connector.bind_connector_metadata(meta)
    if load_result == "timeout":
        with pytest.raises(LMCacheTimeoutError):
            connector.start_load_kv(SimpleNamespace())
        assert calls == []
    else:
        connector.start_load_kv(SimpleNamespace())
        assert calls == ["import", "wait", ("complete", load_result)]
