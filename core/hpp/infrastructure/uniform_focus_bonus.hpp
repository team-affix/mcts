#ifndef UNIFORM_FOCUS_BONUS_HPP
#define UNIFORM_FOCUS_BONUS_HPP

namespace monte_carlo
{

// uniform_focus_bonus
//
// The constant bonus, on the normalized [0, 1] scale, that the focus child's
// score gets at every node.

template<typename IFloat>
struct uniform_focus_bonus
{
    uniform_focus_bonus(IFloat bonus);

    IFloat get_focus_bonus(const auto&) const;

private:
    IFloat bonus_;
};

template<typename IFloat>
uniform_focus_bonus<IFloat>::uniform_focus_bonus(IFloat bonus)
    : bonus_(bonus)
{}

template<typename IFloat>
IFloat uniform_focus_bonus<IFloat>::get_focus_bonus(const auto&) const
{
    return bonus_;
}

}

#endif
