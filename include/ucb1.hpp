#ifndef UCB1_HPP
#define UCB1_HPP

#include <cmath>
#include <cstddef>
#include <limits>

namespace monte_carlo
{

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
struct ucb1
{
    ucb1(IGetVisits&              get_visits,
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
ucb1<INodeHandle, IChoice, IFloat,
     IGetVisits, IGetValue, IWalker,
     IGEC, IGetChoiceCount, IGetChoiceAt>::ucb1(
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
ucb1<INodeHandle, IChoice, IFloat,
     IGetVisits, IGetValue, IWalker,
     IGEC, IGetChoiceCount, IGetChoiceAt>::policy_choose(
        const INodeHandle&     node,
        const IGetChoiceCount& get_choice_count,
        const IGetChoiceAt&    get_choice_at)
{
    IFloat best_score = -std::numeric_limits<IFloat>::infinity();
    size_t best_i     = 0;
    size_t n          = get_choice_count.size();
    IFloat c          = get_exploration_constant_.get_exploration_constant(node);
    IFloat ln_parent  = std::log(static_cast<IFloat>(get_visits_.get_visits(node)));

    for (size_t i = 0; i < n; ++i)
    {
        IChoice           candidate  = get_choice_at.at(i);
        const INodeHandle child_node = walker_.walk(node, candidate);
        size_t            child_v    = get_visits_.get_visits(child_node);

        if (child_v == 0)
        {
            best_score = std::numeric_limits<IFloat>::infinity();
            best_i     = i;
            break;
        }

        IFloat exploit = get_value_.get_value(child_node) / static_cast<IFloat>(child_v);
        IFloat explore = std::sqrt(ln_parent / static_cast<IFloat>(child_v));
        IFloat score   = exploit + c * explore;

        if (score > best_score)
        {
            best_score = score;
            best_i     = i;
        }
    }

    return get_choice_at.at(best_i);
}

}

#endif
