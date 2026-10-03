All problems inherit from a common [`OptimizationProblem`][graphfla.problems.OptimizationProblem] base class, so the calling pattern is uniform: instantiate, optionally call `evaluate(config)` on individual configurations, or call `get_data()` to enumerate all $2^n$ binary configurations and their fitnesses for downstream landscape construction.

!!! warning "Computational cost of `get_data()`"
    `get_data()` enumerates every one of $2^n$ binary configurations. The output has $2^n$ rows — fine for $n \lesssim 20$, prohibitive much beyond that. For large `n`, evaluate individual configurations on demand with `evaluate()` instead.

