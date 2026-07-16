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
    explicit uniform_exploration_constant(IFloat c) : c_(c) {}

    IFloat get_exploration_constant(const auto&) const { return c_; }

private:
    IFloat c_;
};

} // namespace monte_carlo

#endif // UNIFORM_EXPLORATION_CONSTANT_HPP
