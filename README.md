# MathOptLazy.jl

[![Build Status](https://github.com/jump-dev/MathOptLazy.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jump-dev/MathOptLazy.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/jump-dev/MathOptLazy.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jump-dev/MathOptLazy.jl)

[MathOptLazy.jl](https://github.com/jump-dev/MathOptLazy.jl) is an experimental meta-solver for problems with lazy constraints.

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
model = Model(() -> MathOptLazy.Optimizer(HiGHS.Optimizer))
@variable(model, x[1:10] >= 0)
@constraint(model, [i in 1:10], x[i] <= 1, MathOptLazy.Lazy())
```
