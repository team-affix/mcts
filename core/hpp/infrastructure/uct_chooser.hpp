#ifndef UCT_CHOOSER_HPP
#define UCT_CHOOSER_HPP

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
    typename IIsInRollout,
    typename IEnterRollout
>
struct uct_chooser
{
    uct_chooser(IGetVisits&      get_visits,
                IWalker&         walker,
                IPolicyChoose&   policy,
                IRolloutChoose&  rollout,
                IGetCurrentNode& get_current_node,
                ISetCurrentNode& set_current_node,
                IPushNode&       push_node,
                IIsInRollout&    is_in_rollout,
                IEnterRollout&   enter_rollout);

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
    IIsInRollout&    is_in_rollout_;
    IEnterRollout&   enter_rollout_;
};

template<typename INH, typename IC, typename IGVis,
         typename IW, typename IGCC, typename IGCA,
         typename IPC, typename IRC,
         typename IGCN, typename ISCN, typename IPN,
         typename IIIR, typename IER>
uct_chooser<INH, IC, IGVis, IW, IGCC, IGCA, IPC, IRC, IGCN, ISCN, IPN, IIIR, IER>::uct_chooser(
        IGVis& get_visits,
        IW&    walker,
        IPC&   policy,
        IRC&   rollout,
        IGCN&  get_current_node,
        ISCN&  set_current_node,
        IPN&   push_node,
        IIIR&  is_in_rollout,
        IER&   enter_rollout)
    : get_visits_(get_visits)
    , walker_(walker)
    , policy_(policy)
    , rollout_(rollout)
    , get_current_node_(get_current_node)
    , set_current_node_(set_current_node)
    , push_node_(push_node)
    , is_in_rollout_(is_in_rollout)
    , enter_rollout_(enter_rollout)
{}

template<typename INH, typename IC, typename IGVis,
         typename IW, typename IGCC, typename IGCA,
         typename IPC, typename IRC,
         typename IGCN, typename ISCN, typename IPN,
         typename IIIR, typename IER>
IC
uct_chooser<INH, IC, IGVis, IW, IGCC, IGCA, IPC, IRC, IGCN, ISCN, IPN, IIIR, IER>::choose(
        const IGCC& get_choice_count,
        const IGCA& get_choice_at)
{
    INH current = get_current_node_.get_current_node();

    if (is_in_rollout_.is_in_rollout())
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
        enter_rollout_.enter_rollout();

    return chosen;
}

}

#endif
