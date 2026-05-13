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

Base.length(x::_LazyData) = length(x.data)

Base.isempty(x::_LazyData) = isempty(x.data)

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
struct Optimizer{OT} <: MOI.AbstractOptimizer
    inner::OT

    lazy::Dict{Tuple{Type,Type},_LazyData}

    function Optimizer(inner_fn; kwargs...)
        inner = MOI.instantiate(inner_fn; kwargs...)
        return new{typeof(inner)}(inner, Dict{Tuple{Type,Type},_LazyData}())
    end
end

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
    return ret !== nothing && 1 <= ci.value <= length(ret)
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
    return ret === nothing ? 0 : length(ret)
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
    return MOI.ConstraintIndex{F,LazyScalarSet{S}}(length(data))
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

function MOI.optimize!(model::Optimizer)
    needs_solve = true
    x = MOI.get(model, MOI.ListOfVariableIndices())
    while needs_solve
        needs_solve = false
        MOI.optimize!(model.inner)
        if MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
            X = Dict(xi => MOI.get(model, MOI.VariablePrimal(), xi) for xi in x)
            constraints_added = 0
            for v in values(model.lazy)
                constraints_added += _add_if_necessary(model, v, X)
            end
            needs_solve = constraints_added > 0
        end
    end
    return
end

function _add_if_necessary(
    model::Optimizer,
    data::_LazyData,
    x::Dict{MOI.VariableIndex},
)
    constraints_added = 0
    for (i, (f, s)) in enumerate(data.data)
        if data.active[i]
            continue
        end
        y = MOI.Utilities.eval_variables(Base.Fix1(getindex, x), model, f)
        if MOI.Utilities.distance_to_set(y, s) > 0
            data.index[i] = MOI.add_constraint(model.inner, f, s)
            data.active[i] = true
            constraints_added += 1
        end
    end
    return constraints_added
end

end # module MathOptLazy
