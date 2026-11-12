# Selective Qwen3 execution on Ascend

For selective recomputation, use `MPSelectiveBlendConnector` from
`lmcache_ascend.integration.vllm.mp_selective_blend` with the blend server and
**the standard vLLM scheduler**. Do not set `--scheduler-cls` or the removed
`lmcache.mp.blend_skip_first_forward` option. No vLLM/vLLM-Ascend worker patch
is required.

Set `lmcache.mp.blend_recompute_ratio` (default 0.15) and
`lmcache.mp.blend_check_layer` (default 1), use `lmcache_driven` transfer,
`--max-num-seqs 1`, eager synchronous scheduling and disabled prefix caching.
Only dense, unquantized Qwen3 with one DP/PP rank and NHD KV is supported;
LoRA, speculative decoding and lazy offload are excluded.

The connector reports the coverage it will finish through loading and
selective computation. It waits for load events, reaches TP consensus, then
completes that coverage before the ordinary tail forward. Confirmed load
failures recompute the covered suffix; unknown RPC outcomes abort execution.
Coverage includes computed holes and must not be reported as actual cache hits.
A two-question Qwen3-8B/TP1 NPU smoke check completed full recompute and
selective execution at ratios 1.0 and 0.15. Ratio=1 outputs matched full
recompute verbatim. This is a client-path check, not a quality guarantee or
TP2 validation.
