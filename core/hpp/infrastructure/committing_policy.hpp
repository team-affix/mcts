#ifndef COMMITTING_POLICY_HPP
#define COMMITTING_POLICY_HPP

#include <cstddef>

namespace monte_carlo
{

// committing_policy
//
// A selection policy that makes commitments. At the frontier, once the commit
// condition holds for the visits it gathered since becoming the frontier, it
// commits to its most visited child (the first of them on a tie), which becomes
// the new frontier, and returns that choice. Everywhere else, and at the
// frontier until then or while no child has a visit, it chooses through the
// policy it wraps.

template<
    typename INodeHandle,
    typename IChoice,
    typename IGetVisits,
    typename IWalker,
    typename IGetFrontier,
    typename IGetFrontierBaseVisits,
    typename IReachCondition,
    typename ICommit,
    typename IPolicyChoose,
    typename IGetChoiceCount,
    typename IGetChoiceAt
>
struct committing_policy
{
    committing_policy(IGetVisits&             get_visits,
                      IWalker&                walker,
                      IGetFrontier&           get_frontier,
                      IGetFrontierBaseVisits& get_frontier_base_visits,
                      IReachCondition&        reach_condition,
                      ICommit&                commit,
                      IPolicyChoose&          policy);

    IChoice policy_choose(const INodeHandle&     node,
                          const IGetChoiceCount& get_choice_count,
                          const IGetChoiceAt&    get_choice_at);

private:
    IGetVisits&             get_visits_;
    IWalker&                walker_;
    IGetFrontier&           get_frontier_;
    IGetFrontierBaseVisits& get_frontier_base_visits_;
    IReachCondition&        reach_condition_;
    ICommit&                commit_;
    IPolicyChoose&          policy_;
};

template<typename INH, typename IC, typename IGVis, typename IW, typename IGF, typename IGFBV,
         typename IRC, typename ICm, typename IPC, typename IGCC, typename IGCA>
committing_policy<INH, IC, IGVis, IW, IGF, IGFBV, IRC, ICm, IPC, IGCC, IGCA>::committing_policy(
        IGVis& get_visits,
        IW&    walker,
        IGF&   get_frontier,
        IGFBV& get_frontier_base_visits,
        IRC&   reach_condition,
        ICm&   commit,
        IPC&   policy)
    : get_visits_(get_visits)
    , walker_(walker)
    , get_frontier_(get_frontier)
    , get_frontier_base_visits_(get_frontier_base_visits)
    , reach_condition_(reach_condition)
    , commit_(commit)
    , policy_(policy)
{}

template<typename INH, typename IC, typename IGVis, typename IW, typename IGF, typename IGFBV,
         typename IRC, typename ICm, typename IPC, typename IGCC, typename IGCA>
IC committing_policy<INH, IC, IGVis, IW, IGF, IGFBV, IRC, ICm, IPC, IGCC, IGCA>::policy_choose(
        const INH&  node,
        const IGCC& get_choice_count,
        const IGCA& get_choice_at)
{
    const bool at_frontier = node == get_frontier_.frontier();
    if (!at_frontier)
        return policy_.policy_choose(node, get_choice_count, get_choice_at);
    const size_t visits_since = get_visits_.get_visits(node) - get_frontier_base_visits_.frontier_base_visits();
    if (!reach_condition_.reached(visits_since))
        return policy_.policy_choose(node, get_choice_count, get_choice_at);

    size_t best_index  = 0;
    size_t best_visits = 0;
    for (size_t index = 0; index < get_choice_count.size(); ++index)
    {
        const size_t child_visits = get_visits_.get_visits(walker_.walk(node, get_choice_at.at(index)));
        if (child_visits <= best_visits)
            continue;
        best_index  = index;
        best_visits = child_visits;
    }
    if (best_visits == 0)
        return policy_.policy_choose(node, get_choice_count, get_choice_at);

    const IC  chosen = get_choice_at.at(best_index);
    const INH child  = walker_.walk(node, chosen);
    commit_.commit(node, chosen, child, best_visits);
    return chosen;
}

}

#endif
