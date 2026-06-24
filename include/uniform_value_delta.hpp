#ifndef UNIFORM_VALUE_DELTA_HPP
#define UNIFORM_VALUE_DELTA_HPP

namespace monte_carlo
{

// uniform_value_delta<IFloat>
//
// Concrete IGetValueDelta implementation: returns the same value for every node.
// Use with sim::terminate(delta_fn) when all nodes on the backprop path should
// receive an equal share of the episode reward.

template<typename IFloat>
struct uniform_value_delta
{
    uniform_value_delta() : value_(0) {}

    IFloat value()              const { return value_; }
    void   set_value(IFloat v)        { value_ = v;    }

    IFloat get_value_delta(const auto&) const { return value_; }

private:
    IFloat value_;
};

} // namespace monte_carlo

#endif // UNIFORM_VALUE_DELTA_HPP
