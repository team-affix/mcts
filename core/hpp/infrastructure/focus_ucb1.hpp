#ifndef FOCUS_UCB1_HPP
#define FOCUS_UCB1_HPP

#include <cmath>
#include <cstddef>
#include <limits>

namespace monte_carlo
{

// focus_ucb1
//
// ucb1 with a focus: among the children whose normalized mean is within the
// band of the best one (the band shrinks with the node's visits), the first in
// index order is the focus, and its score gets a constant bonus. Children that
// stay tied, such as interchangeable choices, so stop splitting the node's
// visits evenly; the focus's share grows with the node's visits, while every
// other child's exploration term still grows with ln N, so exploration still
// wins now and then. As the band shrinks, any real difference leaves it, and
// the truly best child eventually becomes the focus.
//
// Two passes over the children: the first chooses the first unvisited child if
// there is one, else records every child's mean for normalization and finds
// the best; the second scores each child as normalized mean + c * sqrt(ln N /
// n), plus the bonus for the focus, and returns the first highest score.

template<
    typename INodeHandle,
    typename IChoice,
    typename IFloat,
    typename IGetVisits,
    typename IGetValue,
    typename IIncludeValue,
    typename INormalizeValue,
    typename IWalker,
    typename IGetExplorationConstant,
    typename IGetFocusBand,
    typename IGetFocusBonus,
    typename IGetChoiceCount,
    typename IGetChoiceAt
>
struct focus_ucb1
{
    focus_ucb1(IGetVisits&              get_visits,
               IGetValue&               get_value,
               IIncludeValue&           include_value,
               INormalizeValue&         normalize_value,
               IWalker&                 walker,
               IGetExplorationConstant& get_exploration_constant,
               IGetFocusBand&           get_focus_band,
               IGetFocusBonus&          get_focus_bonus);

    IChoice policy_choose(const INodeHandle&     node,
                          const IGetChoiceCount& get_choice_count,
                          const IGetChoiceAt&    get_choice_at);

private:
    IGetVisits&              get_visits_;
    IGetValue&               get_value_;
    IIncludeValue&           include_value_;
    INormalizeValue&         normalize_value_;
    IWalker&                 walker_;
    IGetExplorationConstant& get_exploration_constant_;
    IGetFocusBand&           get_focus_band_;
    IGetFocusBonus&          get_focus_bonus_;
};

template<typename INH, typename IC, typename IF,
         typename IGVis, typename IGVal, typename IIV, typename INV, typename IW,
         typename IGEC, typename IGFBa, typename IGFBo, typename IGCC, typename IGCA>
focus_ucb1<INH, IC, IF, IGVis, IGVal, IIV, INV, IW, IGEC, IGFBa, IGFBo, IGCC, IGCA>::focus_ucb1(
        IGVis& get_visits,
        IGVal& get_value,
        IIV&   include_value,
        INV&   normalize_value,
        IW&    walker,
        IGEC&  get_exploration_constant,
        IGFBa& get_focus_band,
        IGFBo& get_focus_bonus)
    : get_visits_(get_visits)
    , get_value_(get_value)
    , include_value_(include_value)
    , normalize_value_(normalize_value)
    , walker_(walker)
    , get_exploration_constant_(get_exploration_constant)
    , get_focus_band_(get_focus_band)
    , get_focus_bonus_(get_focus_bonus)
{}

template<typename INH, typename IC, typename IF,
         typename IGVis, typename IGVal, typename IIV, typename INV, typename IW,
         typename IGEC, typename IGFBa, typename IGFBo, typename IGCC, typename IGCA>
IC focus_ucb1<INH, IC, IF, IGVis, IGVal, IIV, INV, IW, IGEC, IGFBa, IGFBo, IGCC, IGCA>::policy_choose(
        const INH&  node,
        const IGCC& get_choice_count,
        const IGCA& get_choice_at)
{
    const size_t choice_count = get_choice_count.size();
    IF           best_mean    = -std::numeric_limits<IF>::infinity();

    for (size_t index = 0; index < choice_count; ++index)
    {
        const INH    child        = walker_.walk(node, get_choice_at.at(index));
        const size_t child_visits = get_visits_.get_visits(child);
        if (child_visits == 0)
            return get_choice_at.at(index);
        const IF mean = get_value_.get_value(child) / static_cast<IF>(child_visits);
        include_value_.include(mean);
        if (mean > best_mean)
            best_mean = mean;
    }

    const size_t parent_visits = get_visits_.get_visits(node);
    const IF     best          = normalize_value_.normalize(best_mean);
    const IF     band          = get_focus_band_.get_band(parent_visits);
    const IF     bonus         = get_focus_bonus_.get_focus_bonus(node);
    const IF     c             = get_exploration_constant_.get_exploration_constant(node);
    const IF     ln_parent     = std::log(static_cast<IF>(parent_visits));
    bool         focus_found   = false;
    IF           best_score    = -std::numeric_limits<IF>::infinity();
    size_t       best_index    = 0;

    for (size_t index = 0; index < choice_count; ++index)
    {
        const INH child        = walker_.walk(node, get_choice_at.at(index));
        const IF  child_visits = static_cast<IF>(get_visits_.get_visits(child));
        const IF  mean         = normalize_value_.normalize(get_value_.get_value(child) / child_visits);
        const IF  explore      = std::sqrt(ln_parent / child_visits);
        const bool is_focus    = !focus_found && best - mean <= band;
        if (is_focus)
            focus_found = true;
        const IF score = mean + c * explore + (is_focus ? bonus : IF(0));
        if (score <= best_score)
            continue;
        best_score = score;
        best_index = index;
    }

    return get_choice_at.at(best_index);
}

}

#endif
