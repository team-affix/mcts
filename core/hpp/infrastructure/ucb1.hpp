#ifndef UCB1_HPP
#define UCB1_HPP

#include <cmath>
#include <cstddef>
#include <limits>

namespace monte_carlo
{

// ucb1
//
// Chooses a child by the UCB1 score: its mean value put on a [0, 1] scale, plus
// c * sqrt(ln(parent visits) / child visits). The scale comes from the lowest
// and highest node means seen anywhere in the tree so far: each selection first
// records its children's means, then normalizes against everything recorded. So
// c means the same thing whatever the rewards' scale, and it may exceed 1 to
// favour exploration. An unvisited child is chosen first, the first of them if
// several; among visited children the first highest score wins.

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
    typename IGetChoiceCount,
    typename IGetChoiceAt
>
struct ucb1
{
    ucb1(IGetVisits&              get_visits,
         IGetValue&               get_value,
         IIncludeValue&           include_value,
         INormalizeValue&         normalize_value,
         IWalker&                 walker,
         IGetExplorationConstant& get_exploration_constant);

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
};

template<typename INH, typename IC, typename IF,
         typename IGVis, typename IGVal, typename IIV, typename INV, typename IW,
         typename IGEC, typename IGCC, typename IGCA>
ucb1<INH, IC, IF, IGVis, IGVal, IIV, INV, IW, IGEC, IGCC, IGCA>::ucb1(
        IGVis& get_visits,
        IGVal& get_value,
        IIV&   include_value,
        INV&   normalize_value,
        IW&    walker,
        IGEC&  get_exploration_constant)
    : get_visits_(get_visits)
    , get_value_(get_value)
    , include_value_(include_value)
    , normalize_value_(normalize_value)
    , walker_(walker)
    , get_exploration_constant_(get_exploration_constant)
{}

template<typename INH, typename IC, typename IF,
         typename IGVis, typename IGVal, typename IIV, typename INV, typename IW,
         typename IGEC, typename IGCC, typename IGCA>
IC ucb1<INH, IC, IF, IGVis, IGVal, IIV, INV, IW, IGEC, IGCC, IGCA>::policy_choose(
        const INH&  node,
        const IGCC& get_choice_count,
        const IGCA& get_choice_at)
{
    const size_t choice_count = get_choice_count.size();

    for (size_t index = 0; index < choice_count; ++index)
    {
        const INH    child        = walker_.walk(node, get_choice_at.at(index));
        const size_t child_visits = get_visits_.get_visits(child);
        if (child_visits == 0)
            return get_choice_at.at(index);
        include_value_.include(get_value_.get_value(child) / static_cast<IF>(child_visits));
    }

    const IF c          = get_exploration_constant_.get_exploration_constant(node);
    const IF ln_parent  = std::log(static_cast<IF>(get_visits_.get_visits(node)));
    IF       best_score = -std::numeric_limits<IF>::infinity();
    size_t   best_index = 0;

    for (size_t index = 0; index < choice_count; ++index)
    {
        const INH    child        = walker_.walk(node, get_choice_at.at(index));
        const IF     child_visits = static_cast<IF>(get_visits_.get_visits(child));
        const IF     exploit      = normalize_value_.normalize(get_value_.get_value(child) / child_visits);
        const IF     explore      = std::sqrt(ln_parent / child_visits);
        const IF     score        = exploit + c * explore;
        if (score <= best_score)
            continue;
        best_score = score;
        best_index = index;
    }

    return get_choice_at.at(best_index);
}

}

#endif
