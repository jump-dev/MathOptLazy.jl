# Copyright (c) 2026 Oscar Dowson, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestMathOptLazy

using JuMP
using Test

import HiGHS
import MathOptInterface as MOI
import MathOptLazy

function runtests()
    is_test(name) = startswith("$name", "test_")
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function test_runtests()
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    config = MOI.Test.Config(; atol = 1e-6)
    MOI.Test.runtests(model, config)
    return
end

function test_jump_cached_knapsack()
    N = 10
    model = Model(() -> MathOptLazy.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x[1:N] >= 0, Int)
    @constraint(model, [i in 1:N], x[i] <= 1, MathOptLazy.Lazy())
    @constraint(model, sum(abs(cos(i)) * x[i] for i in 1:N) <= 0.1 * N)
    @objective(model, Max, sum(abs(sin(i)) * x[i] for i in 1:N))
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    @test primal_status(model) == FEASIBLE_POINT
    @test all(<=(1 + 1e-6), value(x))
    return
end

function test_jump_direct_knapsack()
    N = 10
    model = direct_model(MathOptLazy.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x[1:N] >= 0, Int)
    @constraint(model, [i in 1:N], x[i] <= 1, MathOptLazy.Lazy())
    @constraint(model, sum(abs(cos(i)) * x[i] for i in 1:N) <= 0.1 * N)
    @objective(model, Max, sum(abs(sin(i)) * x[i] for i in 1:N))
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    @test primal_status(model) == FEASIBLE_POINT
    @test all(<=(1 + 1e-6), value(x))
    return
end

function test_jump_direct_basics()
    model = direct_model(MathOptLazy.Optimizer(HiGHS.Optimizer))
    @variable(model, x)
    c = @constraint(model, x <= 1, MathOptLazy.Lazy())
    o = constraint_object(c)
    @test isequal_canonical(o.func, 1.0 * x)
    @test o.set == MathOptLazy.LazyScalarSet(MOI.LessThan(1.0))
    return
end

end  # TestMathOptLazy

TestMathOptLazy.runtests()
