Every problem is a subclass of [`OptimizationProblem`][graphfla.problems.OptimizationProblem] and is used the same way. Create an instance, then call `evaluate(config)` for a single configuration, or `get_data()` to enumerate all $2^n$ binary configurations with their fitness values. The output of `get_data()` can be passed directly to `BooleanLandscape.build_from_data`. Higher fitness is better for every problem.

!!! warning "Cost of full enumeration"
    `get_data()` returns $2^n$ rows, which is practical up to about $n = 20$. For larger $n$, evaluate configurations individually with `evaluate()`, or stream them with `iter_data()`.
