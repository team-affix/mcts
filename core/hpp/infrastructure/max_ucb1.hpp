#ifndef MAX_UCB1_HPP
#define MAX_UCB1_HPP

#include <cmath>
#include <cstddef>
#include <limits>

namespace monte_carlo
{

// max_ucb1
//
// Single-player selection policy: like ucb1, but a child's exploitation term is
// its value as is (the best reward backed up through it, see max_value_update)
// rather than its value per visit. An unvisited child is chosen first; among
// visited children the highest best + c * sqrt(ln(parent visits) / visits)
// wins, the first of them on a tie.

template<
    typename INodeHandle,
    typename IChoice,
    typename IFloat,
    typename IGetVisits,
    typename IGetValue,
    typename IWalker,
    typename IGetExplorationConstant,
    typename IGetChoiceCount,
    typename IGetChoiceAt
>
struct max_ucb1
{
    max_ucb1(IGetVisits&              get_visits,
             IGetValue&               get_value,
             IWalker&                 walker,
             IGetExplorationConstant& get_exploration_constant);

    IChoice policy_choose(const INodeHandle&     node,
                          const IGetChoiceCount& get_choice_count,
                          const IGetChoiceAt&    get_choice_at);

private:
    IGetVisits&              get_visits_;
    IGetValue&               get_value_;
    IWalker&                 walker_;
    IGetExplorationConstant& get_exploration_constant_;
};

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue, typename IWalker,
         typename IGEC, typename IGetChoiceCount, typename IGetChoiceAt>
max_ucb1<INodeHandle, IChoice, IFloat,
         IGetVisits, IGetValue, IWalker,
         IGEC, IGetChoiceCount, IGetChoiceAt>::max_ucb1(
        IGetVisits& get_visits,
        IGetValue&  get_value,
        IWalker&    walker,
        IGEC&       get_exploration_constant)
    : get_visits_(get_visits)
    , get_value_(get_value)
    , walker_(walker)
    , get_exploration_constant_(get_exploration_constant)
{}

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue, typename IWalker,
         typename IGEC, typename IGetChoiceCount, typename IGetChoiceAt>
IChoice
max_ucb1<INodeHandle, IChoice, IFloat,
         IGetVisits, IGetValue, IWalker,
         IGEC, IGetChoiceCount, IGetChoiceAt>::policy_choose(
        const INodeHandle&     node,
        const IGetChoiceCount& get_choice_count,
        const IGetChoiceAt&    get_choice_at)
{
    IFloat best_score   = -std::numeric_limits<IFloat>::infinity();
    size_t best_index   = 0;
    size_t choice_count = get_choice_count.size();
    IFloat c            = get_exploration_constant_.get_exploration_constant(node);
    IFloat ln_parent    = std::log(static_cast<IFloat>(get_visits_.get_visits(node)));

    for (size_t index = 0; index < choice_count; ++index)
    {
        IChoice           candidate    = get_choice_at.at(index);
        const INodeHandle child_node   = walker_.walk(node, candidate);
        size_t            child_visits = get_visits_.get_visits(child_node);

        if (child_visits == 0)
            return candidate;

        IFloat exploit = get_value_.get_value(child_node);
        IFloat explore = std::sqrt(ln_parent / static_cast<IFloat>(child_visits));
        IFloat score   = exploit + c * explore;

        if (score <= best_score)
            continue;

        best_score = score;
        best_index = index;
    }

    return get_choice_at.at(best_index);
}

}

#endif
