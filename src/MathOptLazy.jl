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
    Optimizer(inner_fn; kwargs...)
"""
struct Optimizer{T,OT} <: MOI.AbstractOptimizer
    inner::OT

    saf_gt::_LazyData{MOI.ScalarAffineFunction{T},MOI.GreaterThan{T}}
    saf_lt::_LazyData{MOI.ScalarAffineFunction{T},MOI.LessThan{T}}

    function Optimizer(
        inner_fn;
        coefficient_type::Type{T} = Float64,
        kwargs...,
    ) where {T}
        inner = MOI.instantiate(inner_fn; kwargs...)
        return new{T,typeof(inner)}(
            inner,
            _LazyData{MOI.ScalarAffineFunction{T},MOI.GreaterThan{T}}(),
            _LazyData{MOI.ScalarAffineFunction{T},MOI.LessThan{T}}(),
        )
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

function MOI.get(model::Optimizer, attr::_ATTRIBUTES, arg::Vector{T}) where {T}
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

function MOI.set(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    indices::Vector{<:MOI.VariableIndex},
    args::Vector{T},
) where {T}
    MOI.set.(model, attr, indices, args)
    return
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
    attr::MOI.AbstractConstraintAttribute,
    indices::Vector{<:MOI.ConstraintIndex},
    args::Vector{T},
) where {T}
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

_data(::Optimizer, ::Type{F}, ::Type{S}) where {F,S} = nothing

function _data(
    model::Optimizer{T},
    ::Type{MOI.ScalarAffineFunction{T}},
    ::Type{MOI.GreaterThan{T}},
) where {T}
    return model.saf_gt
end

function _data(
    model::Optimizer{T},
    ::Type{MOI.ScalarAffineFunction{T}},
    ::Type{MOI.LessThan{T}},
) where {T}
    return model.saf_lt
end

function MOI.supports_constraint(
    model::Optimizer,
    ::Type{F},
    ::Type{LazyScalarSet{S}},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    return _data(model, F, S) !== nothing &&
           MOI.supports_constraint(model, F, S)
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{F,LazyScalarSet{S}},
) where {F,S}
    data = _data(model, F, S)
    return data !== nothing && 1 <= ci.value <= length(data)
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{F,LazyScalarSet{S}},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    n = length(_data(model, F, S))
    return [MOI.ConstraintIndex{F,LazyScalarSet{S}}(i) for i in 1:n]
end

function MOI.get(
    model::Optimizer,
    ::MOI.NumberOfConstraints{F,LazyScalarSet{S}},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    return length(_data(model, F, S))
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

function MOI.get(
    model::Optimizer{T},
    ::MOI.ListOfConstraintTypesPresent,
) where {T}
    ret = MOI.get(model.inner, MOI.ListOfConstraintTypesPresent())
    if !isempty(model.saf_gt)
        push!(
            ret,
            (MOI.ScalarAffineFunction{T}, LazyScalarSet{MOI.GreaterThan{T}}),
        )
    end
    if !isempty(model.saf_lt)
        push!(
            ret,
            (MOI.ScalarAffineFunction{T}, LazyScalarSet{MOI.LessThan{T}}),
        )
    end
    return ret
end

function MOI.get(
    model::Optimizer,
    attr::MOI.NumberOfConstraints{F,S},
) where {F<:MOI.AbstractScalarFunction,S<:MOI.AbstractScalarSet}
    n = MOI.get(model.inner, attr)
    if (data = _data(model, F, S)) !== nothing
        n -= sum(data.active)
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

### MOI.optimize!

function MOI.optimize!(model::Optimizer{T}) where {T}
    needs_solve = true
    x = MOI.get(model, MOI.ListOfVariableIndices())
    X = Dict{MOI.VariableIndex,T}(xi => zero(T) for xi in x)
    while needs_solve
        needs_solve = false
        MOI.optimize!(model.inner)
        if MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
            for xi in x
                X[xi] = MOI.get(model, MOI.VariablePrimal(), xi)
            end
            constraints_added =
                _add_if_necessary(model, model.saf_gt, X) +
                _add_if_necessary(model, model.saf_lt, X)
            needs_solve = constraints_added > 0
        end
    end
    return
end

function _add_if_necessary(
    model::Optimizer{T},
    data::_LazyData,
    x::Dict{MOI.VariableIndex,T},
) where {T}
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
