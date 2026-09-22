# Copyright (c) 2026 Oscar Dowson, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestMathOptLazy

using JuMP
using Test

import GLPK
import Gurobi
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
    set_silent(model)
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
    set_silent(model)
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
    MOI.set(model, MOI.Silent(), true)
    config = MOI.Test.Config()
    set = MathOptLazy.LazyScalarSet(inner_set)
    N = MOI.dimension(set)
    x = MOI.add_variables(model, 3)
    constraint_function = function_fn(x)
    @test MOI.output_dimension(constraint_function) == N
    F, S, IS = typeof(constraint_function), typeof(set), typeof(inner_set)
    @test MOI.supports_constraint(model, F, S)
    @test MOI.get(model, MOI.NumberOfConstraints{F,S}()) == 0
    @test isempty(MOI.get(model, MOI.ListOfConstraintIndices{F,S}()))
    @test MOI.get(model, MOI.NumberOfConstraints{F,IS}()) == 0
    @test isempty(MOI.get(model, MOI.ListOfConstraintIndices{F,IS}()))
    c = MOI.add_constraint(model, constraint_function, set)
    if activate
        data = MathOptLazy._data(model, F, IS)
        for (i, (f, s)) in enumerate(data.data)
            data.index[i] = MOI.add_constraint(model.inner, f, s)
            data.status[i] = MathOptLazy._kLAZY_CONSTRAINT_ACTIVE
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
    MOI.set(src, MOI.Silent(), true)
    x = MOI.add_variable(src)
    c = MOI.add_constraint(src, x, MathOptLazy.LazyScalarSet(MOI.ZeroOne()))
    dest = MOI.FileFormats.MOF.Model()
    _ = MOI.copy_to(dest, src)
    contents = sprint(write, dest)
    @test occursin("ZeroOne", contents)
    @test occursin("LazyScalarSet", contents)
    return
end

function test_lazy_bounds()
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
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
    MOI.set(model, MOI.Silent(), true)
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

function _solve_knapsack(optimizer, profit, weight; lazy::Bool)
    N = length(profit)
    model = direct_model(optimizer)
    set_silent(model)
    @variable(model, x[1:N] >= 0, Int)
    @constraint(model, [i in 1:N], x[i] <= 1, MathOptLazy.Lazy(; lazy))
    @constraint(model, weight' * x <= 0.3 * sum(weight))
    @objective(model, Max, profit' * x)
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    @test all(<=(1 + 1e-6), value.(x))
    return objective_value(model)
end

function test_iterative_relax_integrality_knapsack()
    for seed in 1:5
        profit = [1 + abs(sin(seed * i)) for i in 1:20]
        weight = [1 + abs(cos(seed * i)) for i in 1:20]
        expected =
            _solve_knapsack(HiGHS.Optimizer(), profit, weight; lazy = false)
        model = MathOptLazy.Optimizer(HiGHS.Optimizer)
        MOI.set(model, MathOptLazy.Algorithm(), MathOptLazy.Iterative())
        actual = _solve_knapsack(model, profit, weight; lazy = true)
        @test isapprox(actual, expected; atol = 1e-6)
    end
    return
end

function test_iterative_relax_integrality_is_undone()
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 6)
    F = MOI.VariableIndex
    bounds = [
        (x[2], MOI.GreaterThan(-1.0)),
        (x[3], MOI.LessThan(0.5)),
        (x[4], MOI.Interval(-2.0, 2.0)),
        (x[5], MOI.EqualTo(1.0)),
    ]
    for (xi, set) in bounds
        MOI.add_constraint(model, xi, set)
    end
    MOI.add_constraint.(model, x[1:5], MOI.ZeroOne())
    MOI.add_constraint(model, x[6], MOI.Integer())
    MOI.add_constraint(model, x[6], MOI.Interval(0.0, 3.5))
    f = sum(1.0 * xi for xi in x)
    MOI.add_constraint(model, f, MathOptLazy.LazyScalarSet(MOI.LessThan(4.5)))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    types = MOI.get(model.inner, MOI.ListOfConstraintTypesPresent())
    count_of(F, S) = MOI.get(model.inner, MOI.NumberOfConstraints{F,S}())
    before = Dict((F, S) => count_of(F, S) for (F, S) in types)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    # x[3] <= 0.5 and x[6] <= 3.5 are binding only because of integrality.
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 4.0
    @test MOI.get(model, MOI.VariablePrimal(), x[3]) ≈ 0.0
    # The lazy constraint is now in the inner optimizer. Nothing else changed.
    before[(typeof(f), MOI.LessThan{Float64})] = 1
    types = MOI.get(model.inner, MOI.ListOfConstraintTypesPresent())
    @test Dict((F, S) => count_of(F, S) for (F, S) in types) == before
    for (xi, set) in bounds
        ci = MOI.ConstraintIndex{F,typeof(set)}(xi.value)
        @test MOI.get(model, MOI.ConstraintSet(), ci) == set
    end
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

function test_gurobi_solver_specific()
    N = 10
    model = MathOptLazy.Optimizer(Gurobi.Optimizer)
    MOI.set(model, MathOptLazy.Algorithm(), MathOptLazy.SolverSpecific())
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, N)
    MOI.add_constraint.(model, x, MOI.Integer())
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint.(
        model,
        1.0 .* x,
        MathOptLazy.LazyScalarSet(MOI.LessThan(1.0)),
    )
    MOI.add_constraint(
        model,
        sum(abs(cos(i)) * x[i] for i in 1:N),
        MOI.LessThan(0.1 * N),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    f = sum(abs(sin(i)) * x[i] for i in 1:N)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test all(<=(1 + 1e-6), MOI.get(model, MOI.VariablePrimal(), x))
    return
end

function test_glpk_solver_specific()
    N = 10
    model = MathOptLazy.Optimizer(GLPK.Optimizer)
    MOI.set(model, MathOptLazy.Algorithm(), MathOptLazy.SolverSpecific())
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, N)
    MOI.add_constraint.(model, x, MOI.Integer())
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint.(
        model,
        1.0 .* x,
        MathOptLazy.LazyScalarSet(MOI.LessThan(1.0)),
    )
    MOI.add_constraint(
        model,
        sum(abs(cos(i)) * x[i] for i in 1:N),
        MOI.LessThan(0.1 * N),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    f = sum(abs(sin(i)) * x[i] for i in 1:N)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    @test_throws(
        ErrorException(
            """
            The `SolverSpecific` algorithm is not supported by the current solver.

            The current solver type is: `$(GLPK.Optimizer)`

            The supported solvers are:

             * `Gurobi.Optimizer`

            ## Example

            ```julia
            import Gurobi
            import MathOptInterface as MOI
            import MathOptLazy
            optimizer = MOI.OptimizerWithAttributes(
                () -> MathOptLazy.Optimizer(Gurobi.Optimizer),
                MathOptLazy.Algorithm() => MathOptLazy.SolverSpecific(),
            )
            ```
            """,
        ),
        MOI.optimize!(model),
    )
    return
end

function test_delete_bound()
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    c1 = MOI.add_constraint(model, x, MOI.ZeroOne())
    @test MOI.is_valid(model, c1)
    MOI.delete(model, c1)
    @test !MOI.is_valid(model, c1)
    return
end

function test_delete_lazy_inactive()
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.EqualTo(0.0))
    set = MathOptLazy.LazyScalarSet(MOI.EqualTo(1.0))
    c = MOI.add_constraint(model, 1.0 * x, set)
    @test MOI.is_valid(model, c)
    MOI.delete(model, c)
    @test !MOI.is_valid(model, c)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    return
end

function test_delete_lazy_active()
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.EqualTo(0.0))
    set = MathOptLazy.LazyScalarSet(MOI.EqualTo(1.0))
    c = MOI.add_constraint(model, 1.0 * x, set)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.INFEASIBLE
    MOI.delete(model, c)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    return
end

function test_empty()
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    set = MathOptLazy.LazyScalarSet(MOI.EqualTo(1.0))
    c = MOI.add_constraint(model, 1.0 * x, set)
    @test !MOI.is_empty(model)
    @test !MOI.is_empty(model.inner)
    @test !isempty(model.lazy)
    MOI.empty!(model)
    @test MOI.is_empty(model)
    @test MOI.is_empty(model.inner)
    @test isempty(model.lazy)
    return
end

function test_is_valid()
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    set = MathOptLazy.LazyScalarSet(MOI.EqualTo(1.0))
    c = MOI.add_constraint(model, 1.0 * x, set)
    @test !MOI.is_valid(model, typeof(c)(c.value - 1))
    @test MOI.is_valid(model, c)
    @test !MOI.is_valid(model, typeof(c)(c.value + 1))
    F, S = MOI.VariableIndex, MathOptLazy.LazyScalarSet{MOI.ZeroOne}
    @test !MOI.is_valid(model, MOI.ConstraintIndex{F,S}(-1))
    @test !MOI.is_valid(model, MOI.ConstraintIndex{F,S}(0))
    @test !MOI.is_valid(model, MOI.ConstraintIndex{F,S}(1))
    return
end

function test_list_of_constraint_attributes_set()
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    set = MathOptLazy.LazyScalarSet(MOI.EqualTo(1.0))
    c = MOI.add_constraint(model, 1.0 * x[1], set)
    d = MOI.add_constraint(model, 1.0 * x[2], MOI.EqualTo(2.0))
    MOI.set(model, MOI.ConstraintName(), d, "d")
    F, S = MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}
    LS = MathOptLazy.LazyScalarSet{S}
    @test MOI.get(model, MOI.ListOfConstraintAttributesSet{F,S}()) ==
          [MOI.ConstraintName()]
    @test isempty(MOI.get(model, MOI.ListOfConstraintAttributesSet{F,LS}()))
    return
end

function test_relax_integrality_equal_to()
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    c_z = MOI.add_constraint(model, x, MOI.ZeroOne())
    MOI.add_constraint(model, x, MOI.EqualTo(0.5))
    undo = MathOptLazy._relax_integrality(model.inner)
    target = HiGHS.Optimizer()
    MOI.set(target, MOI.Silent(), true)
    y, _ = MOI.add_constrained_variable(target, MOI.EqualTo(0.5))
    @test sprint(print, model.inner) == sprint(print, target)
    undo()
    @test sprint(print, model.inner) != sprint(print, target)
    MOI.add_constraint(target, y, MOI.ZeroOne())
    @test sprint(print, model.inner) == sprint(print, target)
    return
end

function test_relax_integrality_interval()
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    c_z = MOI.add_constraint(model, x, MOI.ZeroOne())
    MOI.add_constraint(model, x, MOI.Interval(-1.0, 2.0))
    undo = MathOptLazy._relax_integrality(model.inner)
    target = HiGHS.Optimizer()
    MOI.set(target, MOI.Silent(), true)
    y, cy = MOI.add_constrained_variable(target, MOI.Interval(0.0, 1.0))
    @test sprint(print, model.inner) == sprint(print, target)
    undo()
    @test sprint(print, model.inner) != sprint(print, target)
    MOI.set(target, MOI.ConstraintSet(), cy, MOI.Interval(-1.0, 2.0))
    MOI.add_constraint(target, y, MOI.ZeroOne())
    @test sprint(print, model.inner) == sprint(print, target)
    return
end

function test_relax_integrality_no_bound()
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    c_z = MOI.add_constraint(model, x, MOI.ZeroOne())
    undo = MathOptLazy._relax_integrality(model.inner)
    target = HiGHS.Optimizer()
    MOI.set(target, MOI.Silent(), true)
    y = MOI.add_variable(target)
    c_l = MOI.add_constraint(target, y, MOI.GreaterThan(0.0))
    c_u = MOI.add_constraint(target, y, MOI.LessThan(1.0))
    @test sprint(print, model.inner) == sprint(print, target)
    undo()
    @test sprint(print, model.inner) != sprint(print, target)
    MOI.delete(target, c_l)
    MOI.delete(target, c_u)
    MOI.add_constraint(target, y, MOI.ZeroOne())
    @test sprint(print, model.inner) == sprint(print, target)
    return
end

function test_relax_integrality_greater_than()
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(-0.5))
    c_z = MOI.add_constraint(model, x, MOI.ZeroOne())
    undo = MathOptLazy._relax_integrality(model.inner)
    target = HiGHS.Optimizer()
    MOI.set(target, MOI.Silent(), true)
    y, cy = MOI.add_constrained_variable(target, MOI.GreaterThan(0.0))
    c_u = MOI.add_constraint(target, y, MOI.LessThan(1.0))
    @test sprint(print, model.inner) == sprint(print, target)
    undo()
    @test sprint(print, model.inner) != sprint(print, target)
    MOI.set(target, MOI.ConstraintSet(), cy, MOI.GreaterThan(-0.5))
    MOI.add_constraint(target, y, MOI.ZeroOne())
    MOI.delete(target, c_u)
    @test sprint(print, model.inner) == sprint(print, target)
    return
end

function test_relax_integrality_less_than()
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.LessThan(1.5))
    c_z = MOI.add_constraint(model, x, MOI.ZeroOne())
    undo = MathOptLazy._relax_integrality(model.inner)
    target = HiGHS.Optimizer()
    MOI.set(target, MOI.Silent(), true)
    y, cy = MOI.add_constrained_variable(target, MOI.LessThan(1.0))
    c_l = MOI.add_constraint(target, y, MOI.GreaterThan(0.0))
    @test sprint(print, model.inner) == sprint(print, target)
    undo()
    @test sprint(print, model.inner) != sprint(print, target)
    MOI.set(target, MOI.ConstraintSet(), cy, MOI.LessThan(1.5))
    MOI.add_constraint(target, y, MOI.ZeroOne())
    MOI.delete(target, c_l)
    @test sprint(print, model.inner) == sprint(print, target)
    return
end

function test_relax_integrality_semi_integer()
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    c_z = MOI.add_constraint(model, x, MOI.ZeroOne())
    t, _ = MOI.add_constrained_variable(model, MOI.Semiinteger(3.0, 4.0))
    undo = MathOptLazy._relax_integrality(model.inner)
    target = HiGHS.Optimizer()
    MOI.set(target, MOI.Silent(), true)
    y = MOI.add_variable(target)
    c_l = MOI.add_constraint(target, y, MOI.GreaterThan(0.0))
    c_u = MOI.add_constraint(target, y, MOI.LessThan(1.0))
    t, _ = MOI.add_constrained_variable(target, MOI.Semiinteger(3.0, 4.0))
    @test sprint(print, model.inner) == sprint(print, target)
    undo()
    @test sprint(print, model.inner) != sprint(print, target)
    MOI.add_constraint(target, y, MOI.ZeroOne())
    MOI.delete(target, c_l)
    MOI.delete(target, c_u)
    @test sprint(print, model.inner) == sprint(print, target)
    return
end

function test_print_log()
    N = 10
    model = MathOptLazy.Optimizer(HiGHS.Optimizer)
    @test MOI.supports(model, MOI.Silent())
    @test MOI.get(model, MOI.Silent()) == false
    MOI.set(model, MOI.Silent(), true)
    @test MOI.get(model, MOI.Silent()) == true
    MOI.set(model, MOI.Silent(), false)
    @test MOI.get(model, MOI.Silent()) == false
    x = MOI.add_variables(model, N)
    for i in 1:N
        MOI.add_constraint(model, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(model, x[i], MOI.Integer())
        set = MathOptLazy.LazyScalarSet(MOI.LessThan(1.0))
        MOI.add_constraint(model, 1.0 * x[i], set)
    end
    f = sum(abs(cos(i)) * x[i] for i in 1:N)
    MOI.add_constraint(model, f, MOI.LessThan(1.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    g = sum(abs(sin(i)) * x[i] for i in 1:N)
    MOI.set(model, MOI.ObjectiveFunction{typeof(g)}(), g)
    dir = mktempdir()
    open(joinpath(dir, "out.log"), "w") do io
        return redirect_stdout(() -> MOI.optimize!(model), io)
    end
    contents = read(joinpath(dir, "out.log"), String)
    for line in [
        "[MathOptLazy] relaxing binary and integer variables",
        "[MathOptLazy] solving current subproblem",
        "[MathOptLazy] added 1 constraints",
        "[MathOptLazy] added 0 constraints",
        "[MathOptLazy] re-enforcing binary and integer variables",
    ]
        @test occursin(line, contents)
    end
    MOI.set(model, MOI.Silent(), true)
    open(joinpath(dir, "out.log"), "w") do io
        return redirect_stdout(() -> MOI.optimize!(model), io)
    end
    @test isempty(read(joinpath(dir, "out.log"), String))
    return
end

end  # TestMathOptLazy

TestMathOptLazy.runtests()
