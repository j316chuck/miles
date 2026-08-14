from examples.tinker_backend.run_tinker_backend import ScriptArgs, _perf_args


def test_dynamic_batching_is_enabled_by_default() -> None:
    args = ScriptArgs(max_tokens_per_gpu=4096)

    perf_args = _perf_args(args)

    assert "--use-dynamic-batch-size" in perf_args
    assert "--max-tokens-per-gpu 4096" in perf_args
    assert "--micro-batch-size" not in perf_args


def test_fixed_batching_avoids_packed_sequences() -> None:
    args = ScriptArgs(use_dynamic_batch_size=False)

    perf_args = _perf_args(args)

    assert "--use-dynamic-batch-size" not in perf_args
    assert "--max-tokens-per-gpu" not in perf_args
    assert "--micro-batch-size 1" in perf_args
