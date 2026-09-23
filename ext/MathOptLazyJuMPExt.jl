# Copyright (c) 2026 Oscar Dowson, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module MathOptLazyJuMPExt

import JuMP
import MathOptInterface as MOI
import MathOptLazy

function JuMP.build_constraint(
    error_fn::Function,
    f::JuMP.AbstractJuMPScalar,
    s::MOI.AbstractScalarSet,
    tag::MathOptLazy.Lazy;
    kwargs...,
)
    if tag.lazy
        s = MathOptLazy.LazyScalarSet(s)
    end
    return JuMP.build_constraint(error_fn, f, s; kwargs...)
end

function JuMP.model_convert(
    model::JuMP.AbstractModel,
    set::MathOptLazy.LazyScalarSet,
)
    return MathOptLazy.LazyScalarSet(JuMP.model_convert(model, set.set))
end

function JuMP.in_set_string(
    mode::MIME"text/plain",
    set::MathOptLazy.LazyScalarSet,
)
    return JuMP.in_set_string(mode, set.set) * " [lazy]"
end

"""
    MathOptLazy.Lazy(
        model::JuMP.GenericModel,
        ::Type{F},
        ::Type{S},
    ) where {F<:JuMP.AbstractJuMPScalar,S<:MOI.AbstractScalarSet}

A constructor for `MathOptLazy.Lazy` that sets the `lazy::Bool` keyword to
`true` if `model` supports converting `F`-in-`S` constraints into lazy
constraints.

!!! compat
    This struct requires JuMP to be loaded to activate the package extension.

## Examples

```julia
julia> using JuMP, MathOptLazy, HiGHS

julia> model = Model(HiGHS.Optimizer);

julia> MathOptLazy.Lazy(model, AffExpr, MOI.LessThan{Float64})
MathOptLazy.Lazy(false)

julia> model = Model(() -> MathOptLazy.Optimizer(HiGHS.Optimizer));

julia> MathOptLazy.Lazy(model, AffExpr, MOI.LessThan{Float64})
MathOptLazy.Lazy(true)
```
"""
function MathOptLazy.Lazy(
    model::JuMP.GenericModel,
    ::Type{F},
    ::Type{S},
) where {F<:JuMP.AbstractJuMPScalar,S<:MOI.AbstractScalarSet}
    lazy = MOI.supports_constraint(
        JuMP.backend(model),
        JuMP.moi_function_type(F),
        MathOptLazy.LazyScalarSet{S},
    )
    return MathOptLazy.Lazy(; lazy)
end

end  # module MathOptLazyJuMPExt
