#ifndef UNIFORM_EXPLORATION_CONSTANT_HPP
#define UNIFORM_EXPLORATION_CONSTANT_HPP

namespace monte_carlo
{

// uniform_exploration_constant<IFloat>
//
// Concrete IGetExplorationConstant implementation: returns the same constant
// for every parent node.  Use when the reward scale is homogeneous across the
// entire tree and a single fixed c suffices for all UCB comparisons.

template<typename IFloat>
struct uniform_exploration_constant
{
    uniform_exploration_constant(IFloat c);

    IFloat get_exploration_constant(const auto&) const;

private:
    IFloat c_;
};

// ---------------------------------------------------------------------------
// member function definitions
// ---------------------------------------------------------------------------

template<typename IFloat>
uniform_exploration_constant<IFloat>::uniform_exploration_constant(IFloat c)
    : c_(c)
{}

template<typename IFloat>
IFloat uniform_exploration_constant<IFloat>::get_exploration_constant(const auto&) const
{
    return c_;
}

} // namespace monte_carlo

#endif // UNIFORM_EXPLORATION_CONSTANT_HPP
