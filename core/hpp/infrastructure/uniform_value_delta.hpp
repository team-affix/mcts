#ifndef UNIFORM_VALUE_DELTA_HPP
#define UNIFORM_VALUE_DELTA_HPP

namespace monte_carlo
{

// uniform_value_delta<IFloat>
//
// Concrete IGetValueDelta implementation: returns the same value for every node.
// Use when all nodes on the backprop path should receive an equal share of the
// episode reward.

template<typename IFloat>
struct uniform_value_delta
{
    uniform_value_delta();

    IFloat value() const;
    void   set_value(IFloat v);

    IFloat get_value_delta(const auto&) const;

private:
    IFloat value_;
};

// ---------------------------------------------------------------------------
// member function definitions
// ---------------------------------------------------------------------------

template<typename IFloat>
uniform_value_delta<IFloat>::uniform_value_delta()
    : value_(0)
{}

template<typename IFloat>
IFloat uniform_value_delta<IFloat>::value() const
{
    return value_;
}

template<typename IFloat>
void uniform_value_delta<IFloat>::set_value(IFloat v)
{
    value_ = v;
}

template<typename IFloat>
IFloat uniform_value_delta<IFloat>::get_value_delta(const auto&) const
{
    return value_;
}

} // namespace monte_carlo

#endif // UNIFORM_VALUE_DELTA_HPP
