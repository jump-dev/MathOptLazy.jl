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

end  # module MathOptLazyJuMPExt
