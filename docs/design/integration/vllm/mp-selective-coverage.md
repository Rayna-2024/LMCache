# MP selective coverage with the standard scheduler

`LMCacheMPSelectiveConnector` in `mp_blend_client.py` follows the legacy
in-process load-spec lifecycle. Ascend's `MPSelectiveBlendConnector` supplies
the Qwen3 selective executor. This PoC supports selective mode only, using
the unmodified vLLM scheduler.

1. Unified lookup returns actual prefix hits and sparse matches.
2. `get_num_new_matched_tokens` saves a `BlendCoverageTask` and returns
   `(coverage_end - computed_prefix_tokens, False)`. The final prompt token
   stays in the ordinary forward. Repeated polls do not advance the request.
3. Standard allocation covers external tokens plus the scheduled tail.
   `update_state_after_alloc` binds the plan to those block IDs;
   `build_connector_meta` attaches it without rewriting the schedule.
4. `start_load_kv` waits on prefix and scatter completion events, including
   events returned with failure. Ascend performs TP consensus, then fills the
   entire promised interval before the ordinary tail forward starts.

Three boundaries have distinct meanings:

- `computed_prefix_tokens`: physically ready before this schedule; never
  overwritten by retrieval or selective execution.
- `prefix_hit_tokens`: actual prefix lookup result. It controls ordinary
  RETRIEVE ranges and the initial STORE watermark, not the sparse coverage.
- `coverage_end`: includes gaps computed by the executor. It is a scheduling
  promise, not a physical-cache hit statistic.

Prefix transfers use the existing device-event-aware RPC future directly.
They do not publish asynchronous receive receipts or invalid-block reports:
this synchronous path resolves confirmed failures before tail execution.
When any TP rank reports a terminal load failure, all ranks recompute the
entire interval after the original ready prefix, using an empty hit set.
An unknown transport outcome or execution error aborts forward; it must never
race full recomputation against a potentially late remote write.

Preempted requests use ordinary full recomputation: recycled physical block IDs
can otherwise look already applied to the server even though their data changed.

The base MP connector retains its prefix bookkeeping and store lifecycle.
Known prefix hit bytes are not falsely extended to cover holes. Remaining
prefetch locks are released by the normal request/session cleanup. KV-only
metadata, model-ready handshakes and scheduler rollback are unnecessary.

CPU checks cover coverage allocation, real RETRIEVE/STORE ranges, completion
ordering, unknown-outcome aborts, TP failure fallback, full-prefill KV equality,
selection budgets and unchanged ready prefixes. After removing the gap-only
client, a Qwen3-8B/TP1 NPU smoke run completed two questions per arm: full
recompute, ratio=1 selective, and ratio=0.15 selective. Ratio=1 outputs matched
full recompute verbatim; both selective arms exercised non-prefix reuse without
load fallback. This smoke check does not establish TP2 or full-dataset accuracy.
