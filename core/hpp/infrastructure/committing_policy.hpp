#ifndef COMMITTING_POLICY_HPP
#define COMMITTING_POLICY_HPP

#include <cstddef>
#include <optional>

namespace monte_carlo
{

// committing_policy
//
// A selection policy that makes commitments. At the frontier, once the commit
// condition holds for the visits it gathered since becoming the frontier, it
// commits to the child the injected finder picks (e.g. the most visited, or the
// highest mean), which becomes the new frontier, and returns that choice.
// Everywhere else, and at the frontier until then or while the finder picks
// nothing, it chooses through the policy it wraps.

template<
    typename INodeHandle,
    typename IChoice,
    typename IGetVisits,
    typename IWalker,
    typename IGetFrontier,
    typename IGetFrontierBaseVisits,
    typename IReachCondition,
    typename IFindCommitChild,
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
                      IFindCommitChild&       find_commit_child,
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
    IFindCommitChild&       find_commit_child_;
    ICommit&                commit_;
    IPolicyChoose&          policy_;
};

template<typename INH, typename IC, typename IGVis, typename IW, typename IGF, typename IGFBV,
         typename IRC, typename IFCC, typename ICm, typename IPC, typename IGCC, typename IGCA>
committing_policy<INH, IC, IGVis, IW, IGF, IGFBV, IRC, IFCC, ICm, IPC, IGCC, IGCA>::committing_policy(
        IGVis& get_visits,
        IW&    walker,
        IGF&   get_frontier,
        IGFBV& get_frontier_base_visits,
        IRC&   reach_condition,
        IFCC&  find_commit_child,
        ICm&   commit,
        IPC&   policy)
    : get_visits_(get_visits)
    , walker_(walker)
    , get_frontier_(get_frontier)
    , get_frontier_base_visits_(get_frontier_base_visits)
    , reach_condition_(reach_condition)
    , find_commit_child_(find_commit_child)
    , commit_(commit)
    , policy_(policy)
{}

template<typename INH, typename IC, typename IGVis, typename IW, typename IGF, typename IGFBV,
         typename IRC, typename IFCC, typename ICm, typename IPC, typename IGCC, typename IGCA>
IC committing_policy<INH, IC, IGVis, IW, IGF, IGFBV, IRC, IFCC, ICm, IPC, IGCC, IGCA>::policy_choose(
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

    const std::optional<IC> found = find_commit_child_.find(node, get_choice_count, get_choice_at);
    if (!found.has_value())
        return policy_.policy_choose(node, get_choice_count, get_choice_at);

    const IC  chosen = found.value();
    const INH child  = walker_.walk(node, chosen);
    commit_.commit(node, chosen, child, get_visits_.get_visits(child));
    return chosen;
}

}

#endif
