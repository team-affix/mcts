#ifndef DBUCT_CHOOSER_HPP
#define DBUCT_CHOOSER_HPP

#include <algorithm>
#include <cstddef>
#include "value_objects/dbuct_frame.hpp"

namespace monte_carlo
{

template<
    typename INodeHandle,
    typename IChoice,
    typename IGetVisits,
    typename IGetGrant,
    typename IForestep,
    typename IGetTopFrame,
    typename IWalker,
    typename IGetChoiceCount,
    typename IGetChoiceAt,
    typename IPolicyChoose,
    typename IRolloutChoose,
    typename IIsInRollout,
    typename IEnterRollout
>
struct dbuct_chooser
{
    dbuct_chooser(IGetVisits&     get_visits,
                  IGetGrant&      get_grant,
                  IForestep&      forestep,
                  IGetTopFrame&   get_top_frame,
                  IWalker&        walker,
                  IPolicyChoose&  policy,
                  IRolloutChoose& rollout,
                  IIsInRollout&   is_in_rollout,
                  IEnterRollout&  enter_rollout);

    IChoice choose(const IGetChoiceCount& get_choice_count,
                   const IGetChoiceAt&    get_choice_at);

private:
    IGetVisits&     get_visits_;
    IGetGrant&      get_grant_;
    IForestep&      forestep_;
    IGetTopFrame&   get_top_frame_;
    IWalker&        walker_;
    IPolicyChoose&  policy_;
    IRolloutChoose& rollout_;
    IIsInRollout&   is_in_rollout_;
    IEnterRollout&  enter_rollout_;
};

template<typename INH, typename IC, typename IGVis, typename IGG,
         typename IFo, typename IGTF,
         typename IW, typename IGCC, typename IGCA,
         typename IPC, typename IRC,
         typename IIIR, typename IER>
dbuct_chooser<INH, IC, IGVis, IGG, IFo, IGTF, IW, IGCC, IGCA, IPC, IRC, IIIR, IER>::dbuct_chooser(
        IGVis& get_visits,
        IGG&   get_grant,
        IFo&   forestep,
        IGTF&  get_top_frame,
        IW&    walker,
        IPC&   policy,
        IRC&   rollout,
        IIIR&  is_in_rollout,
        IER&   enter_rollout)
    : get_visits_(get_visits)
    , get_grant_(get_grant)
    , forestep_(forestep)
    , get_top_frame_(get_top_frame)
    , walker_(walker)
    , policy_(policy)
    , rollout_(rollout)
    , is_in_rollout_(is_in_rollout)
    , enter_rollout_(enter_rollout)
{}

template<typename INH, typename IC, typename IGVis, typename IGG,
         typename IFo, typename IGTF,
         typename IW, typename IGCC, typename IGCA,
         typename IPC, typename IRC,
         typename IIIR, typename IER>
IC
dbuct_chooser<INH, IC, IGVis, IGG, IFo, IGTF, IW, IGCC, IGCA, IPC, IRC, IIIR, IER>::choose(
        const IGCC& get_choice_count,
        const IGCA& get_choice_at)
{
    if (is_in_rollout_.is_in_rollout())
        return rollout_.rollout_choose(get_choice_count, get_choice_at);

    dbuct_frame<INH>& current = get_top_frame_.top();

    IC  chosen       = policy_.policy_choose(current.handle, get_choice_count, get_choice_at);
    INH child_handle = walker_.walk(current.handle, chosen);

    size_t remaining_budget = current.budget - current.visit_lump;
    size_t grant_k = std::min(get_grant_.get_grant(current.handle), remaining_budget);

    forestep_.forestep(dbuct_frame<INH>(child_handle, grant_k));

    size_t child_visits = get_visits_.get_visits(child_handle);

    if (child_visits == 0)
        enter_rollout_.enter_rollout();

    return chosen;
}

}

#endif
