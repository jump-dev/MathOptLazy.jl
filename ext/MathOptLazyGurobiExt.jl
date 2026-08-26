# Copyright (c) 2026 Oscar Dowson, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module MathOptLazyGurobiExt

import Gurobi
import MathOptInterface as MOI
import MathOptLazy

function _add_constraints!(
    model::MathOptLazy.Optimizer{Gurobi.Optimizer},
    data::MathOptLazy._LazyData,
)
    for (i, (f, s)) in enumerate(data.data)
        if !data.active[i]
            c = data.index[i] = MOI.add_constraint(model.inner, f, s)
            MOI.set(model.inner, Gurobi.ConstraintAttribute("Lazy"), c, 1)
            data.active[i] = true
        end
    end
    return
end

function MathOptLazy._optimize!(
    model::MathOptLazy.Optimizer{Gurobi.Optimizer},
    ::MathOptLazy.SolverSpecific,
)
    for data in values(model.lazy)
        _add_constraints!(model, data)
    end
    MOI.optimize!(model.inner)
    return
end

end  # module MathOptLazyJuMPExt
