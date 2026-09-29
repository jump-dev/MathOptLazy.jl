# MathOptLazy.jl

[![Build Status](https://github.com/jump-dev/MathOptLazy.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jump-dev/MathOptLazy.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/jump-dev/MathOptLazy.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jump-dev/MathOptLazy.jl)

[MathOptLazy.jl](https://github.com/jump-dev/MathOptLazy.jl) is a meta-solver
for problems with lazy constraints.

## License

`MathOptLazy.jl` is licensed under the [MIT License](https://github.com/jump-dev/MultiObjectiveAlgorithms.jl/blob/main/LICENSE.md).

## Getting help

If you need help, please ask a question on the [JuMP community forum](https://jump.dev/forum).

If you have a reproducible example of a bug, please [open a GitHub issue](https://github.com/jump-dev/MathOptLazy.jl/issues/new).

## Installation

Install `MathOptLazy` using `Pkg.add`:

```julia
import Pkg
Pkg.add("MathOptLazy")
```

## Use with JuMP

Use `MathOptLazy.jl` with JuMP as follows:
```julia
using JuMP
import HiGHS
import MathOptLazy
# Pass () -> MathOptLazy.Optimizer(inner_optimizer) as the solver
model = Model(() -> MathOptLazy.Optimizer(HiGHS.Optimizer))
# Choose an algorithm
set_attribute(model, MathOptLazy.Algorithm(), MathOptLazy.Iterative())
@variable(model, x[1:10] >= 0)
# Tag constraints as lazy
@constraint(model, [i in 1:10], x[i] <= 1, MathOptLazy.Lazy())
# You can also pass the `lazy` keyword to Lazy()
is_lazy = rand(Bool)
@constraint(model, sum(x) <= 3, MathOptLazy.Lazy(; lazy = is_lazy))
# You can also use this constructor to opt-in to lazy constraints of the given
# type if and only if the solver supports them. This simplifies writing a model
# where the user gets to choose the solver.
tag = MathOptLazy.Lazy(model, AffExpr, MOI.GreaterThan{Float64})
@constraint(model, sum(x) >= 2, tag)
```

## Algorithm

Control the algorithm used to handle the lazy constraints by setting the
`MathOptLazy.Algorithm` attribute. See the docstring for details. The supoprted
values are:

 * `MathOptLazy.Iterative()` [default]
 * `MathOptLazy.Callback()`
 * `MathOptLazy.SolverSpecific()`

See their docstrings for details.
