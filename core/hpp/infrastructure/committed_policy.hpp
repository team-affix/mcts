#ifndef COMMITTED_POLICY_HPP
#define COMMITTED_POLICY_HPP

#include <optional>

namespace monte_carlo
{

// committed_policy
//
// A selection policy that follows commitments: at a committed node it returns
// the committed choice, and elsewhere it chooses through the policy it wraps.

template<
    typename INodeHandle,
    typename IChoice,
    typename IFindCommitment,
    typename IPolicyChoose,
    typename IGetChoiceCount,
    typename IGetChoiceAt
>
struct committed_policy
{
    committed_policy(IFindCommitment& find_commitment,
                     IPolicyChoose&   policy);

    IChoice policy_choose(const INodeHandle&     node,
                          const IGetChoiceCount& get_choice_count,
                          const IGetChoiceAt&    get_choice_at);

private:
    IFindCommitment& find_commitment_;
    IPolicyChoose&   policy_;
};

template<typename INH, typename IC, typename IFC, typename IPC, typename IGCC, typename IGCA>
committed_policy<INH, IC, IFC, IPC, IGCC, IGCA>::committed_policy(
        IFC& find_commitment,
        IPC& policy)
    : find_commitment_(find_commitment)
    , policy_(policy)
{}

template<typename INH, typename IC, typename IFC, typename IPC, typename IGCC, typename IGCA>
IC committed_policy<INH, IC, IFC, IPC, IGCC, IGCA>::policy_choose(
        const INH&  node,
        const IGCC& get_choice_count,
        const IGCA& get_choice_at)
{
    const std::optional<IC> committed = find_commitment_.find_commitment(node);
    if (committed.has_value())
        return committed.value();
    return policy_.policy_choose(node, get_choice_count, get_choice_at);
}

}

#endif
