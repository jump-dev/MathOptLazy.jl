# Copyright (c) 2026 Oscar Dowson, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module MathOptLazy

import MathOptInterface as MOI

### LazyScalarSet

struct LazyScalarSet{S<:MOI.AbstractScalarSet} <: MOI.AbstractScalarSet
    set::S
end

function MOI.Utilities.supports_shift_constant(
    ::Type{LazyScalarSet{S}},
) where {S<:MOI.AbstractScalarSet}
    return MOI.Utilities.supports_shift_constant(S)
end

function MOI.Utilities.shift_constant(set::LazyScalarSet, constant)
    return LazyScalarSet(MOI.Utilities.shift_constant(set.set, constant))
end

# This doesn't fit the Schema, but we do it anyway.
function MOI.FileFormats.MOF.moi_to_object(
    set::LazyScalarSet,
    x::Dict{MOI.VariableIndex,String},
)
    inner_set = MOI.FileFormats.MOF.moi_to_object(set.set, x)
    return (; type = "LazyScalarSet", set = inner_set)
end

function MOI.FileFormats.MOF.set_to_moi(
    ::Val{:LazyScalarSet},
    ::Type{T},
    object::Dict,
) where {T}
    return LazyScalarSet(MOI.FileFormats.MOF.set_to_moi(T, object["set"]))
end

"""
    Lazy(; lazy::Bool = true)

A struct to pass as a positional argument to `JuMP.@constraint` so we can
intercept `JuMP.build_constraint` and turn the constriant into a lazy constraint.

The struct has an inner keyword argument so that you can pass `Lazy()`, but also
`Lazy(; lazy)` if you want a simple way to choose whether to make the constraint
lazy at runtime.

!!! compat
    This struct requires JuMP to be loaded to activate the package extension.

## Examples

```julia
julia> using JuMP, MathOptLazy

julia> model = Model();

julia> @variable(model, x);

julia> @constraint(model, x >= 1, MathOptLazy.Lazy())
x ≥ 1 [lazy]

julia> @constraint(model, x >= 1, MathOptLazy.Lazy(; lazy = false))
x ≥ 1
```
"""
struct Lazy
    lazy::Bool

    Lazy(; lazy::Bool = true) = new(lazy)
end

Base.broadcastable(::Lazy) = Ref(Lazy())

### _LazyData

@enum(
    _LAZY_CONSTRAINT_STATUS,
    _kLAZY_CONSTRAINT_ACTIVE,
    _kLAZY_CONSTRAINT_INACTIVE,
    _kLAZY_CONSTRAINT_DELETED,
)

struct _LazyData{F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    data::Vector{Tuple{F,S}}
    status::Vector{_LAZY_CONSTRAINT_STATUS}
    index::Vector{MOI.ConstraintIndex{F,S}}

    function _LazyData{
        F,
        S,
    }() where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
        return new{F,S}(
            Tuple{F,S}[],
            _LAZY_CONSTRAINT_STATUS[],
            MOI.ConstraintIndex{F,S}[],
        )
    end
end

### Algorithm

"""
    Algorithm() <: MOI.AbstractOptimizerAttribute

An `MOI.AbstractOptimizerAttribute` to control which algorithm we use to solve
the lazy constraints.

Supported values are

 * `Iterative()` [default]
 * `Callback()`
 * `SolverSpecific()`
"""
struct Algorithm <: MOI.AbstractOptimizerAttribute end

abstract type AbstractAlgorithm end

"""
    Iterative()

This algorithm iteratively solves a sequence of problems that iteratively add
violated lazy constraints to the main problem.

This algorithm works for all problem types, including continuous problems with
no discrete variables.

The downside is that it may not re-use information between solves. For example,
when solving a MIP a solver like HiGHS re-uses only the primal start; it does
not re-use information from the branch-and-bound tree.

In an attempt to improve performance, the algorithm runs in two phases. The
first phase iterates on the continuous relaxation. The second phase restores
integrality and iterates on the original problem, starting from the lazy
constraints that the first phase added. This is helpful when the optimal MIP
solution is close to the optimal LP relaxation.

The first phase is cheap because a solver that uses the simplex method can
warm-start from the previous basis after new constraints are added, whereas
every iteration of the second phase solves a mixed-integer program from
scratch. The second phase is still needed because a lazy constraint can be
violated by an integer solution even if the relaxation satisfies it.
"""
struct Iterative <: AbstractAlgorithm end

"""
    Callback()

This algorithm uses a `MOI.LazyConstraintCallback` to add violated lazy
constraints to the main problem.

This algorithm works only for problems with discrete variables and only if the
solver supports `MOI.LazyConstraintCallback`.
"""
struct Callback <: AbstractAlgorithm end

"""
    SolverSpecific()

This algorithm uses a solver-specific extension to add lazy constraints.

The following solvers are supported:

 * `Gurobi.jl`
"""
struct SolverSpecific <: AbstractAlgorithm end

### Optimizer

"""
    Optimizer(inner_fn; kwargs...) <: MOI.AbstractOptimizer

Construct a new optimizer that wraps the result of
`MOI.instantiate(inner_fn; kwargs...)`.

## Example

```julia
julia> import MathOptLazy

julia> import HiGHS

julia> model = MathOptLazy.Optimizer(HiGHS.Optimizer)
MathOptLazy.Optimizer{Float64, HiGHS.Optimizer}
├ ObjectiveSense: FEASIBILITY_SENSE
├ ObjectiveFunctionType: MOI.ScalarAffineFunction{Float64}
├ NumberOfVariables: 0
└ NumberOfConstraints: 0

julia> model = MathOptLazy.Optimizer(HiGHS.Optimizer; with_bridge_type = Float64)
MathOptLazy.Optimizer{Float64, MOIB.LazyBridgeOptimizer{HiGHS.Optimizer}}
├ ObjectiveSense: FEASIBILITY_SENSE
├ ObjectiveFunctionType: MOI.ScalarAffineFunction{Float64}
├ NumberOfVariables: 0
└ NumberOfConstraints: 0
```
"""
mutable struct Optimizer{OT<:MOI.ModelLike} <: MOI.AbstractOptimizer
    inner::OT
    algorithm::AbstractAlgorithm
    lazy::Dict{Tuple{Type,Type},_LazyData}
    silent::Bool
    # Result attributes
    barrier_iterations::Int64
    node_count::Int64
    simplex_iterations::Int64
    solve_time_sec::Float64

    function Optimizer(inner_fn; kwargs...)
        inner = MOI.instantiate(inner_fn; kwargs...)
        return new{typeof(inner)}(
            inner,
            Iterative(),
            Dict{Tuple{Type,Type},_LazyData}(),
            false,
            0,
            0,
            0,
            NaN,
        )
    end
end

### Algorithm

MOI.supports(::Optimizer, ::Algorithm) = true

MOI.get(model::Optimizer, ::Algorithm) = model.algorithm

function MOI.set(model::Optimizer, ::Algorithm, value::AbstractAlgorithm)
    model.algorithm = value
    return
end

MOI.Utilities.map_indices(::Function, algorithm::AbstractAlgorithm) = algorithm

### MOI.Silent

MOI.supports(::Optimizer, ::MOI.Silent) = true

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

function MOI.set(model::Optimizer, ::MOI.Silent, value::Bool)
    model.silent = value
    if MOI.supports(model.inner, MOI.Silent())
        MOI.set(model.inner, MOI.Silent(), value)
    end
    return
end

### Fallbacks

function MOI.empty!(model::Optimizer)
    MOI.empty!(model.inner)
    empty!(model.lazy)
    model.barrier_iterations = 0
    model.node_count = 0
    model.simplex_iterations = 0
    model.solve_time_sec = NaN
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.inner) && isempty(model.lazy)
end

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(dest, src)
end

### Generic fallbacks
###
### These cover most cases. Some are specifically overloaded below because of
### method ambiguities.

const _ATTRIBUTES = Union{
    MOI.AbstractConstraintAttribute,
    MOI.AbstractModelAttribute,
    MOI.AbstractOptimizerAttribute,
    MOI.AbstractVariableAttribute,
}

function MOI.set(model::Optimizer, attr::_ATTRIBUTES, args...)
    return MOI.set(model.inner, attr, args...)
end

function MOI.get(model::Optimizer, attr::_ATTRIBUTES, args...)
    return MOI.get(model.inner, attr, args...)
end

function MOI.get(model::Optimizer, attr::_ATTRIBUTES, arg::Vector)
    return MOI.get.(model, attr, arg)
end

### MOI.AbstractOptimizerAttribute

function MOI.supports(model::Optimizer, arg::MOI.AbstractOptimizerAttribute)
    return MOI.supports(model.inner, arg)
end

function MOI.set(model::Optimizer, attr::MOI.AbstractOptimizerAttribute, value)
    MOI.set(model.inner, attr, value)
    return
end

function MOI.get(model::Optimizer, attr::MOI.AbstractOptimizerAttribute)
    return MOI.get(model.inner, attr)
end

### AbstractModelAttribute

function MOI.supports(model::Optimizer, arg::MOI.AbstractModelAttribute)
    return MOI.supports(model.inner, arg)
end

function MOI.get(model::Optimizer, attr::MOI.ListOfModelAttributesSet)
    return MOI.get(model.inner, attr)
end

### AbstractVariableAttribute

function MOI.is_valid(model::Optimizer, x::MOI.VariableIndex)
    return MOI.is_valid(model.inner, x)
end

function MOI.supports(
    model::Optimizer,
    arg::MOI.AbstractVariableAttribute,
    ::Type{MOI.VariableIndex},
)
    return MOI.supports(model.inner, arg, MOI.VariableIndex)
end

### AbstractConstraintAttribute

function MOI.is_valid(model::Optimizer, ci::MOI.ConstraintIndex)
    return MOI.is_valid(model.inner, ci)
end

function MOI.supports(
    model::Optimizer,
    arg::MOI.AbstractConstraintAttribute,
    ::Type{MOI.ConstraintIndex{F,S}},
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet}
    return MOI.supports(model.inner, arg, MOI.ConstraintIndex{F,S})
end

function MOI.set(
    model::Optimizer,
    attr::Union{MOI.AbstractConstraintAttribute,MOI.AbstractVariableAttribute},
    indices::Vector,
    args::Vector,
)
    MOI.set.(model, attr, indices, args)
    return
end

### Get by String name

function MOI.get(model::Optimizer, ::Type{MOI.VariableIndex}, args...)
    return MOI.get(model.inner, MOI.VariableIndex, args...)
end

function MOI.get(model::Optimizer, T::Type{<:MOI.ConstraintIndex}, args...)
    return MOI.get(model.inner, T, args...)
end

### MOI.add_variable

MOI.add_variable(model::Optimizer) = MOI.add_variable(model.inner)

MOI.add_variables(model::Optimizer, n::Int) = MOI.add_variables(model.inner, n)

### MOI.add_constraint

function MOI.supports_constraint(
    model::Optimizer,
    F::Type{<:MOI.AbstractFunction},
    S::Type{<:MOI.AbstractSet},
)
    return MOI.supports_constraint(model.inner, F, S)
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.AbstractFunction,
    s::MOI.AbstractSet,
)
    return MOI.add_constraint(model.inner, f, s)
end

### MOI.delete!

function MOI.delete(model::Optimizer, ci::MOI.ConstraintIndex{F,S}) where {F,S}
    MOI.delete(model.inner, ci)
    return
end

function MOI.delete(
    model::Optimizer,
    ci::MOI.ConstraintIndex{F,LazyScalarSet{S}},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    data = _data(model, F, S)
    if data.status[ci.value] == _kLAZY_CONSTRAINT_ACTIVE
        MOI.delete(model.inner, data.index[ci.value])
        data.status[ci.value] = _kLAZY_CONSTRAINT_DELETED
    elseif data.status[ci.value] == _kLAZY_CONSTRAINT_INACTIVE
        data.status[ci.value] = _kLAZY_CONSTRAINT_DELETED
    end
    return
end

### MOI.compute_conflict!

MOI.compute_conflict!(model::Optimizer) = MOI.compute_conflict!(model.inner)

### LazyConstraints

function _maybe_data(
    model::Optimizer,
    ::Type{F},
    ::Type{S},
)::Union{Nothing,_LazyData{F,S}} where {F,S}
    return get(model.lazy, (F, S), nothing)
end

function _data(model::Optimizer, ::Type{F}, ::Type{S}) where {F,S}
    return get!(_LazyData{F,S}, model.lazy, (F, S))
end

function MOI.supports_constraint(
    model::Optimizer,
    ::Type{F},
    ::Type{LazyScalarSet{S}},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    return MOI.supports_constraint(model, F, S)
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{F,LazyScalarSet{S}},
) where {F,S}
    ret = _maybe_data(model, F, S)
    if ret == nothing
        return false
    end
    status = get(ret.status, ci.value, _kLAZY_CONSTRAINT_DELETED)
    return status != _kLAZY_CONSTRAINT_DELETED
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{F,LazyScalarSet{S}},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    ret = MOI.ConstraintIndex{F,LazyScalarSet{S}}[]
    if (data = _maybe_data(model, F, S)) !== nothing
        for (i, status) in enumerate(data.status)
            if status != _kLAZY_CONSTRAINT_DELETED
                push!(ret, MOI.ConstraintIndex{F,LazyScalarSet{S}}(i))
            end
        end
    end
    return ret
end

function MOI.get(
    model::Optimizer,
    ::MOI.NumberOfConstraints{F,LazyScalarSet{S}},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    if (data = _maybe_data(model, F, S)) !== nothing
        return count(!=(_kLAZY_CONSTRAINT_DELETED), data.status)
    end
    return 0
end

function MOI.add_constraint(
    model::Optimizer,
    f::F,
    s::LazyScalarSet{S},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    data = _data(model, F, S)
    push!(data.data, (f, s.set))
    push!(data.status, _kLAZY_CONSTRAINT_INACTIVE)
    push!(data.index, MOI.ConstraintIndex{F,S}(0))
    return MOI.ConstraintIndex{F,LazyScalarSet{S}}(length(data.data))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    ci::MOI.ConstraintIndex{F,LazyScalarSet{S}},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    return _data(model, F, S).data[ci.value][1]
end

function MOI.get(
    model::Optimizer,
    ::MOI.CanonicalConstraintFunction,
    ci::MOI.ConstraintIndex{F,LazyScalarSet{S}},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    return MOI.Utilities.canonical(MOI.get(model, MOI.ConstraintFunction(), ci))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{F,LazyScalarSet{S}},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    return LazyScalarSet(_data(model, F, S).data[ci.value][2])
end

function MOI.get(model::Optimizer, ::MOI.ListOfConstraintTypesPresent)
    ret = MOI.get(model.inner, MOI.ListOfConstraintTypesPresent())
    for (F, S) in keys(model.lazy)
        push!(ret, (F, LazyScalarSet{S}))
    end
    return ret
end

function MOI.get(
    model::Optimizer,
    attr::MOI.NumberOfConstraints{F,S},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    n = MOI.get(model.inner, attr)
    if (data = _maybe_data(model, F, S)) !== nothing
        for status in data.status
            n -= status == _kLAZY_CONSTRAINT_ACTIVE
        end
    end
    return n
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ListOfConstraintIndices{F,S},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    ret = MOI.get(model.inner, attr)
    if (data = _maybe_data(model, F, S)) !== nothing
        in_model = Set{MOI.ConstraintIndex{F,S}}(
            ci for (ci, status) in zip(data.index, data.status) if
            status == _kLAZY_CONSTRAINT_ACTIVE
        )
        ret = filter!(ci -> !(ci in in_model), ret)
    end
    return ret
end

function MOI.get(
    ::Optimizer,
    ::MOI.ListOfConstraintAttributesSet{F,LazyScalarSet{S}},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    return MOI.AbstractConstraintAttribute[]
end

function MOI.supports(
    model::Optimizer,
    ::MOI.AbstractConstraintAttribute,
    ::Type{MOI.ConstraintIndex{F,LazyScalarSet{S}}},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    return false
end

function MOI.get(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ::MOI.ConstraintIndex{F,LazyScalarSet{S}},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    msg = "This attribute is not supported for lazy constraints"
    return throw(MOI.GetAttributeNotAllowed(attr, msg))
end

function MOI.set(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ::MOI.ConstraintIndex{F,LazyScalarSet{S}},
    value::Any,
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    msg = "This attribute is not supported for lazy constraints"
    return throw(MOI.SetAttributeNotAllowed(attr, msg))
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{F,LazyScalarSet{S}},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    data = _data(model, F, S)
    if data.status[ci.value] == _kLAZY_CONSTRAINT_ACTIVE
        return MOI.get(model.inner, attr, data.index[ci.value])
    end
    f = first(data.data[ci.value])
    return MOI.Utilities.eval_variables(model, f) do xi
        return MOI.get(model, MOI.VariablePrimal(attr.result_index), xi)
    end
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{F,LazyScalarSet{S}},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    data = _data(model, F, S)
    if data.status[ci.value] == _kLAZY_CONSTRAINT_ACTIVE
        return MOI.get(model.inner, attr, data.index[ci.value])
    end
    # If a lazy constraint is not active, it must be non-binding, and its dual
    # is therefore zero.
    return 0.0
end

### MOI.BarrierIterations

MOI.get(model::Optimizer, ::MOI.BarrierIterations) = model.barrier_iterations

### MOI.NodeCount

MOI.get(model::Optimizer, ::MOI.NodeCount) = model.node_count

### MOI.SimplexIterations

MOI.get(model::Optimizer, ::MOI.SimplexIterations) = model.simplex_iterations

### MOI.SolveTimeSec

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time_sec

### MathOptLazy.NumberOfConstraintsActive

"""
    NumberOfConstraintsActive{F,LazyScalarSet{S}}() where {
        F<:MOI.AbstractScalarFunction,
        S<:MOI.AbstractScalarSet,
    }

Return a count of the number of constraints that are active in the subproblem.
"""
struct NumberOfConstraintsActive{
    F<:MOI.AbstractScalarFunction,
    S<:LazyScalarSet{<:MOI.AbstractScalarSet},
} <: MOI.AbstractModelAttribute end

MOI.is_set_by_optimize(::NumberOfConstraintsActive) = true

function MOI.get(
    model::Optimizer,
    ::NumberOfConstraintsActive{F,LazyScalarSet{S}},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    return count(==(_kLAZY_CONSTRAINT_ACTIVE), _data(model, F, S).status)
end

### MOI.optimize!

function _try_get(model::Optimizer, attr::MOI.AbstractModelAttribute, default)
    try
        return MOI.get(model.inner, attr)
    catch
        return default
    end
end

function _optimize_inner!(model::Optimizer)
    MOI.optimize!(model.inner)
    model.barrier_iterations += _try_get(model, MOI.BarrierIterations(), 0)
    model.node_count += _try_get(model, MOI.NodeCount(), 0)
    model.simplex_iterations += _try_get(model, MOI.SimplexIterations(), 0)
    return
end

function MOI.optimize!(model::Optimizer)
    start_time = time()
    model.barrier_iterations = 0
    model.node_count = 0
    model.simplex_iterations = 0
    model.solve_time_sec = NaN
    if isempty(model.lazy)
        _optimize_inner!(model)
    else
        _optimize!(model, model.algorithm)
    end
    model.solve_time_sec = time() - start_time
    return
end

### MathOptLazy.Iterative

function _optimize!(model::Optimizer, ::Iterative)
    if (undo = _relax_integrality(model.inner)) !== nothing
        if !model.silent
            println("[MathOptLazy] relaxing binary and integer variables")
        end
        _iterate(model; start = false)
        if !model.silent
            println("[MathOptLazy] re-enforcing binary and integer variables")
        end
        undo()
    end
    _iterate(model; start = true)
    return
end

function _iterate(model::Optimizer; start::Bool)
    constraints_added = 1  # A white lie to enter the iteration loop
    x = MOI.get(model, MOI.ListOfVariableIndices())
    # TODO(odow): if the solver supports VariablePrimalStart, we will update the
    # primal starts during the solve process. This is destructive and in-place.
    # That is, we will overwrite any start set by the user. We can't restore the
    # start at the end of the loop because this may invalidate the inner
    # solver's solution. To fix properly, we should cache the start in
    # ::Optimizer, but this is a hassle and no one probably cares. Revisit this
    # decision if it ever becomes a problem.
    start &= MOI.supports(model, MOI.VariablePrimalStart(), MOI.VariableIndex)
    while constraints_added > 0
        constraints_added = 0
        if !model.silent
            println("[MathOptLazy] solving current subproblem\n")
        end
        _optimize_inner!(model)
        if MOI.get(model, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
            # The problem is unbounded, but it might not be if we add more
            # constraints.
            for v in values(model.lazy)
                constraints_added += _add_if_unbounded(model, v)
            end
        elseif MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
            X = Dict(xi => MOI.get(model, MOI.VariablePrimal(), xi) for xi in x)
            for v in values(model.lazy)
                constraints_added += _add_if_feasible(model, v, X)
            end
            if start && constraints_added > 0
                for (xi, v) in X
                    MOI.set(model, MOI.VariablePrimalStart(), xi, v)
                end
            end
        end
        if !model.silent
            println("\n[MathOptLazy] added $(constraints_added) constraints")
        end
    end
    return
end

# Guess-timate the coefficient type based on variable bounds.
function _coefficient_type(model::MOI.ModelLike)
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        if F == MOI.VariableIndex && (T = _coefficient_type(S)) !== nothing
            return T
        end
    end
    return Float64
end

_coefficient_type(::Type{S}) where {S} = nothing

_coefficient_type(::Type{MOI.GreaterThan{T}}) where {T} = T

_coefficient_type(::Type{MOI.LessThan{T}}) where {T} = T

_coefficient_type(::Type{MOI.EqualTo{T}}) where {T} = T

_coefficient_type(::Type{MOI.Interval{T}}) where {T} = T

function _relax_integrality(model::MOI.ModelLike)
    F = MOI.VariableIndex
    integer_ci = MOI.get(model, MOI.ListOfConstraintIndices{F,MOI.Integer}())
    binary_ci = MOI.get(model, MOI.ListOfConstraintIndices{F,MOI.ZeroOne}())
    if isempty(integer_ci) && isempty(binary_ci)
        return nothing
    end
    integer_x = MOI.get.(model, MOI.ConstraintFunction(), integer_ci)
    MOI.delete(model, integer_ci)
    binary_x = MOI.get.(model, MOI.ConstraintFunction(), binary_ci)
    ret = _relax_binaries(model, binary_x)
    function undo()
        MOI.add_constraint.(model, integer_x, MOI.Integer())
        MOI.add_constraint.(model, binary_x, MOI.ZeroOne())
        for (xi, set) in ret
            _update_set!(model, xi, set)
        end
        return
    end
    return undo
end

function _get_set(model, x::MOI.VariableIndex, ::Type{S}) where {S}
    ci = MOI.ConstraintIndex{MOI.VariableIndex,S}(x.value)
    if MOI.is_valid(model, ci)
        return MOI.get(model, MOI.ConstraintSet(), ci)
    end
    return nothing
end

function _update_set!(model, x::MOI.VariableIndex, set::S) where {S}
    ci = MOI.ConstraintIndex{MOI.VariableIndex,S}(x.value)
    MOI.set(model, MOI.ConstraintSet(), ci, set)
    return
end

function _update_set!(model, x::MOI.VariableIndex, ::Type{S}) where {S}
    MOI.delete(model, MOI.ConstraintIndex{MOI.VariableIndex,S}(x.value))
    return
end

function _relax_binaries(model::MOI.ModelLike, x::Vector{MOI.VariableIndex})
    ret = Any[]
    T = _coefficient_type(model)
    for xi in x
        _relax_binaries(ret, model, xi, T)
    end
    return ret
end

function _relax_binaries(
    ret::Vector{Any},
    model::MOI.ModelLike,
    x::MOI.VariableIndex,
    ::Type{T},
) where {T}
    ci = MOI.ConstraintIndex{MOI.VariableIndex,MOI.ZeroOne}(x.value)
    MOI.delete(model, ci)
    if _get_set(model, x, MOI.EqualTo{T}) !== nothing
        return
    elseif (set = _get_set(model, x, MOI.Interval{T})) !== nothing
        new_set = MOI.Interval(max(set.lower, zero(T)), min(set.upper, one(T)))
        _update_set!(model, x, new_set)
        push!(ret, (x, set))
        return
    end
    if (set = _get_set(model, x, MOI.GreaterThan{T})) !== nothing
        _update_set!(model, x, MOI.GreaterThan(max(set.lower, zero(T))))
        push!(ret, (x, set))
    else
        MOI.add_constraint(model, x, MOI.GreaterThan(zero(T)))
        push!(ret, (x, MOI.GreaterThan{T}))
    end
    if (set = _get_set(model, x, MOI.LessThan{T})) !== nothing
        _update_set!(model, x, MOI.LessThan(min(set.upper, one(T))))
        push!(ret, (x, set))
    else
        MOI.add_constraint(model, x, MOI.LessThan(one(T)))
        push!(ret, (x, MOI.LessThan{T}))
    end
    return
end

function _add_if_unbounded(model::Optimizer, data::_LazyData)
    # Strategy: add 1/3 of the total constraints. This is arbitrary. If a model
    # is unbounded with lazy constraints, it's not a great model, and it has a
    # high likelihood that it requires _all_ the constraints to be added. I'm
    # imagining something like 0 <= x <= Lazy(1) in a knapsack problem.
    n = div(length(data.data), 3, RoundUp)
    constraints_added = 0
    for (i, (f, s)) in enumerate(data.data)
        if constraints_added >= n
            break
        elseif data.status[i] == _kLAZY_CONSTRAINT_INACTIVE
            data.index[i] = MOI.add_constraint(model.inner, f, s)
            data.status[i] = _kLAZY_CONSTRAINT_ACTIVE
            constraints_added += 1
        end
    end
    return constraints_added
end

function _add_if_feasible(
    model::Optimizer,
    data::_LazyData,
    x::Dict{MOI.VariableIndex},
)
    constraints_added = 0
    for (i, (f, s)) in enumerate(data.data)
        if data.status[i] != _kLAZY_CONSTRAINT_INACTIVE
            continue
        end
        y = MOI.Utilities.eval_variables(Base.Fix1(getindex, x), model, f)
        if MOI.Utilities.distance_to_set(y, s) > 0
            data.index[i] = MOI.add_constraint(model.inner, f, s)
            data.status[i] = _kLAZY_CONSTRAINT_ACTIVE
            constraints_added += 1
        end
    end
    return constraints_added
end

### MathOptLazy.Callback

function _optimize!(model::Optimizer, ::Callback)
    function callback(cb_data)
        x = MOI.get(model, MOI.ListOfVariableIndices())
        X = Dict(
            xi => MOI.get(model.inner, MOI.CallbackVariablePrimal(cb_data), xi) for xi in x
        )
        # We don't check `.is_active` in this loop because callbacks are weird.
        # In some solvers, callbacks may be called at a point that was
        # previously cut off because the added cut was later removed. The only
        # guarantee is that the solver won't terminate until this loop produces
        # no new cuts.
        for data in values(model.lazy)
            for (i, (f, s)) in enumerate(data.data)
                y = MOI.Utilities.eval_variables(
                    Base.Fix1(getindex, X),
                    model.inner,
                    f,
                )
                if MOI.Utilities.distance_to_set(y, s) > 0
                    MOI.submit(model.inner, MOI.LazyConstraint(cb_data), f, s)
                end
            end
        end
        return
    end
    MOI.set(model.inner, MOI.LazyConstraintCallback(), callback)
    _optimize_inner!(model)
    return
end

### MathOptLazy.SolverSpecific

function _optimize!(model::Optimizer{T}, ::SolverSpecific) where {T}
    return error(
        """
        The `SolverSpecific` algorithm is not supported by the current solver.

        The current solver type is: `$T`

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
    )
end

end # module MathOptLazy
