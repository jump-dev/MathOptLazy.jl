# Copyright (c) 2026 Oscar Dowson, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestMathOptLazy

using JuMP
using Test

import GLPK
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
    @constraint(model, c[i in 1:N], x[i] <= 1, MathOptLazy.Lazy())
    @test endswith(sprint(show, c[1]), " [lazy]")
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

function test_jump_broadcast()
    model = Model(() -> MathOptLazy.Optimizer(HiGHS.Optimizer))
    @variable(model, x[1:3])
    c = @constraint(model, x .<= 1:3, MathOptLazy.Lazy())
    @test c isa Vector && length(c) == 3
    for (i, ci) in enumerate(c)
        @test constraint_object(ci).set ==
              MathOptLazy.LazyScalarSet(MOI.LessThan{Float64}(i))
    end
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

function _basic_constraint_test_helper(
    function_fn::Function,
    inner_set::MOI.AbstractScalarSet;
    activate::Bool,
)
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    config = MOI.Test.Config()
    set = MathOptLazy.LazyScalarSet(inner_set)
    N = MOI.dimension(set)
    x = MOI.add_variables(model, 3)
    constraint_function = function_fn(x)
    @test MOI.output_dimension(constraint_function) == N
    F, S, IS = typeof(constraint_function), typeof(set), typeof(inner_set)
    @test MOI.supports_constraint(model, F, S)
    @test MOI.get(model, MOI.NumberOfConstraints{F,S}()) == 0
    c = MOI.add_constraint(model, constraint_function, set)
    if activate
        data = MathOptLazy._data(model, F, IS)
        for (i, (f, s)) in enumerate(data.data)
            data.index[i] = MOI.add_constraint(model.inner, f, s)
            data.active[i] = true
        end
    end
    c_inner = MOI.add_constraint(model, constraint_function, inner_set)
    @test MOI.get(model, MOI.NumberOfConstraints{F,S}()) == 1
    @test MOI.get(model, MOI.NumberOfConstraints{F,IS}()) == 1
    @test MOI.get(model, MOI.ListOfConstraintIndices{F,S}()) == [c]
    @test MOI.get(model, MOI.ListOfConstraintIndices{F,IS}()) == [c_inner]
    @test (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
    @test MOI.is_valid(model, c)
    @test !MOI.is_valid(model, typeof(c)(c.value + 1)) isa Bool
    @test !MOI.is_valid(model, typeof(c)(c.value - 1)) isa Bool
    @test !MOI.is_valid(model, typeof(c)(c.value + 12345))
    # Don't compare directly, because `f` might not be canonicalized.
    f = MOI.get(model, MOI.ConstraintFunction(), c)
    @test isapprox(f, constraint_function, config)
    cf = MOI.get(model, MOI.CanonicalConstraintFunction(), c)
    @test isapprox(cf, constraint_function, config)
    @test MOI.get(model, MOI.ConstraintSet(), c) == set
    MOI.add_constraints(
        model,
        [constraint_function, constraint_function],
        [set, set],
    )
    @test MOI.get(model, MOI.NumberOfConstraints{F,S}()) == 3
    @test length(MOI.get(model, MOI.ListOfConstraintIndices{F,S}())) == 3
    c_indices = MOI.get(model, MOI.ListOfConstraintIndices{F,S}())
    @test all(MOI.is_valid.(model, c_indices))
    MOI.set(model, MOI.ConstraintName(), [c_inner], ["c"])
    @test MOI.get(model, MOI.ConstraintName(), [c_inner]) == ["c"]
    @test_throws(
        MOI.SetAttributeNotAllowed,
        MOI.set(model, MOI.ConstraintName(), [c], ["c"]),
    )
    @test_throws(
        MOI.GetAttributeNotAllowed,
        MOI.get(model, MOI.ConstraintName(), [c]),
    )
    return
end

function test_basic_scalaraffinefunction_greaterthan()
    _basic_constraint_test_helper(MOI.GreaterThan(1.0); activate = true) do x
        return sum(sin(i) * x[i] for i in 1:length(x))
    end
    _basic_constraint_test_helper(MOI.GreaterThan(1.0); activate = false) do x
        return sum(sin(i) * x[i] for i in 1:length(x))
    end
    return
end

function test_basic_scalaraffinefunction_lessthan()
    _basic_constraint_test_helper(MOI.LessThan(1.0); activate = true) do x
        return sum(sin(i) * x[i] for i in 1:length(x))
    end
    _basic_constraint_test_helper(MOI.LessThan(1.0); activate = false) do x
        return sum(sin(i) * x[i] for i in 1:length(x))
    end
    return
end

function test_writing_mof_file()
    src = MathOptLazy.Optimizer(HiGHS.Optimizer)
    x = MOI.add_variable(src)
    c = MOI.add_constraint(src, x, MathOptLazy.LazyScalarSet(MOI.ZeroOne()))
    dest = MOI.FileFormats.MOF.Model()
    _ = MOI.copy_to(dest, src)
    contents = sprint(write, dest)
    @test occursin("ZeroOne", contents)
    @test !occursin("LazyScalarSet", contents)
    return
end

function test_lazy_bounds()
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    x = MOI.add_variable(model)
    set = MathOptLazy.LazyScalarSet(MOI.GreaterThan(0.0))
    MOI.add_constraint(model, x, set)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    f = 1.0 * x
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(model, MOI.VariablePrimal(), x) == 0.0
    return
end

function test_lazy_bounds_knapsack()
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    x = MOI.add_variables(model, 22)
    set = MathOptLazy.LazyScalarSet(MOI.GreaterThan(0.0))
    MOI.add_constraint.(model, x, set)
    set = MathOptLazy.LazyScalarSet(MOI.LessThan(1.0))
    MOI.add_constraint.(model, x, set)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    f = rand(22)' * x
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    return
end

function test_jump_glpk_callback()
    N = 10
    model = Model(() -> MathOptLazy.Optimizer(GLPK.Optimizer))
    opt = unsafe_backend(model)
    @test MOI.supports(opt, MathOptLazy.Algorithm())
    @test MOI.get(opt, MathOptLazy.Algorithm()) == MathOptLazy.Iterative()
    set_attribute(model, MathOptLazy.Algorithm(), MathOptLazy.Callback())
    @test MOI.get(opt, MathOptLazy.Algorithm()) == MathOptLazy.Callback()
    set_silent(model)
    @variable(model, x[1:N] >= 0, Int)
    @constraint(model, c[i in 1:N], x[i] <= 1, MathOptLazy.Lazy())
    @test endswith(sprint(show, c[1]), " [lazy]")
    @constraint(model, sum(abs(cos(i)) * x[i] for i in 1:N) <= 0.1 * N)
    @objective(model, Max, sum(abs(sin(i)) * x[i] for i in 1:N))
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    @test primal_status(model) == FEASIBLE_POINT
    @test all(<=(1 + 1e-6), value(x))
    return
end

end  # TestMathOptLazy

TestMathOptLazy.runtests()
