#ifndef SIM_CHOOSER_HPP
#define SIM_CHOOSER_HPP

#include <cstddef>

namespace monte_carlo
{

template<
    typename INodeHandle,
    typename IChoice,
    typename IGetVisits,
    typename IWalker,
    typename IGetChoiceCount,
    typename IGetChoiceAt,
    typename IPolicyChoose,
    typename IRolloutChoose,
    typename IGetCurrentNode,
    typename ISetCurrentNode,
    typename IPushNode,
    typename IGetInRollout,
    typename ISetInRollout
>
struct sim_chooser
{
    sim_chooser(IGetVisits&      get_visits,
                IWalker&         walker,
                IPolicyChoose&   policy,
                IRolloutChoose&  rollout,
                IGetCurrentNode& get_current_node,
                ISetCurrentNode& set_current_node,
                IPushNode&       push_node,
                IGetInRollout&   get_in_rollout,
                ISetInRollout&   set_in_rollout);

    IChoice choose(const IGetChoiceCount& get_choice_count,
                   const IGetChoiceAt&    get_choice_at);

private:
    IGetVisits&      get_visits_;
    IWalker&         walker_;
    IPolicyChoose&   policy_;
    IRolloutChoose&  rollout_;
    IGetCurrentNode& get_current_node_;
    ISetCurrentNode& set_current_node_;
    IPushNode&       push_node_;
    IGetInRollout&   get_in_rollout_;
    ISetInRollout&   set_in_rollout_;
};

template<typename INH, typename IC, typename IGVis,
         typename IW, typename IGCC, typename IGCA,
         typename IPC, typename IRC,
         typename IGCN, typename ISCN, typename IPN,
         typename IGIR, typename ISIR>
sim_chooser<INH, IC, IGVis, IW, IGCC, IGCA, IPC, IRC, IGCN, ISCN, IPN, IGIR, ISIR>::sim_chooser(
        IGVis& get_visits,
        IW&    walker,
        IPC&   policy,
        IRC&   rollout,
        IGCN&  get_current_node,
        ISCN&  set_current_node,
        IPN&   push_node,
        IGIR&  get_in_rollout,
        ISIR&  set_in_rollout)
    : get_visits_(get_visits)
    , walker_(walker)
    , policy_(policy)
    , rollout_(rollout)
    , get_current_node_(get_current_node)
    , set_current_node_(set_current_node)
    , push_node_(push_node)
    , get_in_rollout_(get_in_rollout)
    , set_in_rollout_(set_in_rollout)
{}

template<typename INH, typename IC, typename IGVis,
         typename IW, typename IGCC, typename IGCA,
         typename IPC, typename IRC,
         typename IGCN, typename ISCN, typename IPN,
         typename IGIR, typename ISIR>
IC
sim_chooser<INH, IC, IGVis, IW, IGCC, IGCA, IPC, IRC, IGCN, ISCN, IPN, IGIR, ISIR>::choose(
        const IGCC& get_choice_count,
        const IGCA& get_choice_at)
{
    INH current = get_current_node_.get_current_node();

    if (get_in_rollout_.get_in_rollout())
    {
        IC chosen = rollout_.rollout_choose(get_choice_count, get_choice_at);
        set_current_node_.set_current_node(walker_.walk(current, chosen));
        return chosen;
    }

    IC  chosen = policy_.policy_choose(current, get_choice_count, get_choice_at);
    INH child  = walker_.walk(current, chosen);

    push_node_.push(child);
    set_current_node_.set_current_node(child);

    size_t child_visits = get_visits_.get_visits(child);

    if (child_visits == 0)
        set_in_rollout_.set_in_rollout(true);

    return chosen;
}

}

#endif
