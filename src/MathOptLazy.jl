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

function MOI.FileFormats.MOF.moi_to_object(
    set::LazyScalarSet,
    x::Dict{MOI.VariableIndex,String},
)
    return MOI.FileFormats.MOF.moi_to_object(set.set, x)
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

struct _LazyData{F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    data::Vector{Tuple{F,S}}
    active::Vector{Bool}
    index::Vector{MOI.ConstraintIndex{F,S}}

    function _LazyData{
        F,
        S,
    }() where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
        return new{F,S}(Tuple{F,S}[], Bool[], MOI.ConstraintIndex{F,S}[])
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
"""
struct Algorithm <: MOI.AbstractOptimizerAttribute end

abstract type AbstractAlgorithm end

"""
    Iterative()

This algorithm iteratively solves a sequence of problems that iteratively add
violated lazy constraints to the main problem.

This algorithm works for all problem types, including continuous problems with
no discrete variables. The downside is that it may not re-use information
between solves.
"""
struct Iterative <: AbstractAlgorithm end

"""
    Callback()

This algorithm uses a `MOI.LazyConstraintCallback` to add violated laz
 constraints to the main problem.

This algorithm works only for problems with discrete variables and only if the
solver supports `MOI.LazyConstraintCallback`.
"""
struct Callback <: AbstractAlgorithm end

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

    function Optimizer(inner_fn; kwargs...)
        inner = MOI.instantiate(inner_fn; kwargs...)
        return new{typeof(inner)}(
            inner,
            Iterative(),
            Dict{Tuple{Type,Type},_LazyData}(),
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

### Fallbacks

function MOI.empty!(model::Optimizer)
    MOI.empty!(model.inner)
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.inner)
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
    return ret !== nothing && 1 <= ci.value <= length(ret.data)
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{F,LazyScalarSet{S}},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    n = MOI.get(model, MOI.NumberOfConstraints{F,LazyScalarSet{S}}())
    return [MOI.ConstraintIndex{F,LazyScalarSet{S}}(i) for i in 1:n]
end

function MOI.get(
    model::Optimizer,
    ::MOI.NumberOfConstraints{F,LazyScalarSet{S}},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    ret = _maybe_data(model, F, S)
    return ret === nothing ? 0 : length(ret.data)
end

function MOI.add_constraint(
    model::Optimizer,
    f::F,
    s::LazyScalarSet{S},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    data = _data(model, F, S)
    push!(data.data, (f, s.set))
    push!(data.active, false)
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
        n -= sum((data).active)
    end
    return n
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ListOfConstraintIndices{F,S},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    ret = MOI.get(model.inner, attr)
    if (data = _data(model, F, S)) !== nothing
        in_model = Set(ci for (ci, z) in zip(data.index, data.active) if z)
        ret = filter!(ci -> !(ci in in_model), ret)
    end
    return ret
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

### MOI.optimize!

MOI.optimize!(model::Optimizer) = _optimize!(model, model.algorithm)

function _optimize!(model::Optimizer, ::Iterative)
    needs_solve = true
    x = MOI.get(model, MOI.ListOfVariableIndices())
    # TODO(odow): if the solver supports VariablePrimalStart, we will update the
    # primal starts during the solve process. This is destructive and in-place.
    # That is, we will overwrite any start set by the user. We can't restore the
    # start at the end of the loop because this may invalidate the inner
    # solver's solution. To fix properly, we should cache the start in
    # ::Optimizer, but this is a hassle and no one probably cares. Revisit this
    # decision if it ever becomes a problem.
    start = MOI.supports(model, MOI.VariablePrimalStart(), MOI.VariableIndex)
    while needs_solve
        needs_solve = false
        MOI.optimize!(model.inner)
        if MOI.get(model, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
            # The problem is unbounded, but it might not be if we add more
            # constraints.
            for v in values(model.lazy)
                needs_solve |= _add_if_unbounded(model, v)
            end
        elseif MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
            X = Dict(xi => MOI.get(model, MOI.VariablePrimal(), xi) for xi in x)
            for v in values(model.lazy)
                needs_solve |= _add_if_feasible(model, v, X)
            end
            if start && needs_solve
                for (xi, v) in X
                    MOI.set(model, MOI.VariablePrimalStart(), xi, v)
                end
            end
        end
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
        elseif !data.active[i]
            data.index[i] = MOI.add_constraint(model.inner, f, s)
            data.active[i] = true
            constraints_added += 1
        end
    end
    return constraints_added > 0
end

function _add_if_feasible(
    model::Optimizer,
    data::_LazyData,
    x::Dict{MOI.VariableIndex},
)
    needs_solve = false
    for (i, (f, s)) in enumerate(data.data)
        if data.active[i]
            continue
        end
        y = MOI.Utilities.eval_variables(Base.Fix1(getindex, x), model, f)
        if MOI.Utilities.distance_to_set(y, s) > 0
            data.index[i] = MOI.add_constraint(model.inner, f, s)
            data.active[i] = true
            needs_solve = true
        end
    end
    return needs_solve
end

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
    MOI.optimize!(model.inner)
    return
end

end # module MathOptLazy
