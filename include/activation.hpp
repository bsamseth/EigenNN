#pragma once

#include <Eigen/Dense>
#include "definitions.hpp"

namespace activation {

template<typename F, typename DF, typename DDF>
class ActivationFunction {
    public:
        template<typename Derived>
        auto evaluate(const Eigen::DenseBase<Derived>& x) const {
            return x.unaryExpr(F{});
        }

        template<typename Derived>
        auto derivative(const Eigen::DenseBase<Derived>& y) const {
            return y.unaryExpr(DF{});
        }

        template<typename Derived>
        auto dblDerivative(const Eigen::DenseBase<Derived>& y) const {
            return y.unaryExpr(DDF{});
        }
};

namespace functors {

namespace relu {
struct eval {
    Real operator() (Real x) const {
        return x > 0 ? x : 0;
    }
};
struct deriv {
    Real operator() (Real y) const {
        return y > 0 ? 1 : 0;
    }
};
struct dblDeriv {
    Real operator() (Real y) const {
        (void) y;
        return 0;
    }
};
}

namespace identity {
struct eval {
    Real operator() (Real x) const {
        return x;
    }
};
struct deriv {
    Real operator() (Real y) const {
        (void) y;
        return 1;
    }
};
struct dblDeriv {
    Real operator() (Real y) const {
        (void) y;
        return 0;
    }
};
}

namespace sigmoid {
struct eval {
    Real operator() (Real x) const {
        return 1 / (1 + std::exp(-x));
    }
};
struct deriv {
    Real operator() (Real y) const {
        return y * (1 - y);
    }
};
struct dblDeriv {
    Real operator() (Real y) const {
        return y * (1 - y) * (1 - 2*y);
    }
};
}
}


using Relu = ActivationFunction<functors::relu::eval, functors::relu::deriv, functors::relu::dblDeriv>;
using Identity = ActivationFunction<functors::identity::eval, functors::identity::deriv, functors::identity::dblDeriv>;
using Sigmoid = ActivationFunction<functors::sigmoid::eval, functors::sigmoid::deriv, functors::sigmoid::dblDeriv>;

}

