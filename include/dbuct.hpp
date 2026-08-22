#ifndef DBUCT_HPP
#define DBUCT_HPP

#include <algorithm>
#include <cstddef>
#include "dbuct_frame.hpp"

namespace monte_carlo
{

template<
    typename INodeHandle,
    typename IChoice,
    typename IGetVisits,
    typename IGetDispatches,
    typename ISetDispatches,
    typename IComputeBatchSize,
    typename IForestep,
    typename IBackstep,
    typename IGetTopFrame,
    typename IWalker,
    typename IGetChoiceCount,
    typename IGetChoiceAt,
    typename IPolicyChoose,
    typename IRolloutChoose,
    typename ITerminate
>
struct dbuct
{
    dbuct(IGetVisits&        get_visits,
          IGetDispatches&    get_dispatches,
          ISetDispatches&    set_dispatches,
          IComputeBatchSize& compute_batch_size,
          IForestep&         forestep,
          IBackstep&         backstep,
          IGetTopFrame&      get_top_frame,
          IWalker&           walker,
          IPolicyChoose&     policy,
          IRolloutChoose&    rollout,
          ITerminate&        terminate);

    IChoice choose(const IGetChoiceCount& get_choice_count,
                   const IGetChoiceAt&    get_choice_at);

    void terminate_and_backtrack();

    bool in_rollout() const;

private:
    IGetVisits&        get_visits_;
    IGetDispatches&    get_dispatches_;
    ISetDispatches&    set_dispatches_;
    IComputeBatchSize& compute_batch_size_;
    IForestep&         forestep_;
    IBackstep&         backstep_;
    IGetTopFrame&      get_top_frame_;
    IWalker&           walker_;
    IPolicyChoose&     policy_;
    IRolloutChoose&    rollout_;
    ITerminate&        terminate_;

    bool in_rollout_;
};

template<typename INH, typename IC, typename IGVis,
         typename IGD, typename ISD, typename IBS,
         typename IFo, typename IBa, typename IGTF,
         typename IW, typename IGCC, typename IGCA,
         typename IPC, typename IRC, typename IT>
dbuct<INH, IC, IGVis, IGD, ISD, IBS, IFo, IBa, IGTF, IW, IGCC, IGCA, IPC, IRC, IT>::dbuct(
        IGVis& get_visits,
        IGD&   get_dispatches,
        ISD&   set_dispatches,
        IBS&   compute_batch_size,
        IFo&   forestep,
        IBa&   backstep,
        IGTF&  get_top_frame,
        IW&    walker,
        IPC&   policy,
        IRC&   rollout,
        IT&    terminate)
    : get_visits_(get_visits)
    , get_dispatches_(get_dispatches)
    , set_dispatches_(set_dispatches)
    , compute_batch_size_(compute_batch_size)
    , forestep_(forestep)
    , backstep_(backstep)
    , get_top_frame_(get_top_frame)
    , walker_(walker)
    , policy_(policy)
    , rollout_(rollout)
    , terminate_(terminate)
    , in_rollout_(false)
{}

template<typename INH, typename IC, typename IGVis,
         typename IGD, typename ISD, typename IBS,
         typename IFo, typename IBa, typename IGTF,
         typename IW, typename IGCC, typename IGCA,
         typename IPC, typename IRC, typename IT>
IC
dbuct<INH, IC, IGVis, IGD, ISD, IBS, IFo, IBa, IGTF, IW, IGCC, IGCA, IPC, IRC, IT>::choose(
        const IGCC& get_choice_count,
        const IGCA& get_choice_at)
{
    if (in_rollout_)
        return rollout_.rollout_choose(get_choice_count, get_choice_at);

    dbuct_frame<INH>& current = get_top_frame_.top();

    IC  chosen       = policy_.policy_choose(current.handle, get_choice_count, get_choice_at);
    INH child_handle = walker_.walk(current.handle, chosen);

    size_t current_dispatches = get_dispatches_.get_dispatches(current.handle);
    size_t remaining_budget   = current.budget - current.visit_lump;
    size_t grant_k = std::min(
        compute_batch_size_.compute_batch_size(current_dispatches),
        remaining_budget);
    set_dispatches_.set_dispatches(current.handle, current_dispatches + 1);

    forestep_.forestep(dbuct_frame<INH>(child_handle, grant_k));

    size_t child_visits = get_visits_.get_visits(child_handle);

    if (child_visits == 0)
        in_rollout_ = true;

    return chosen;
}

template<typename INH, typename IC, typename IGVis,
         typename IGD, typename ISD, typename IBS,
         typename IFo, typename IBa, typename IGTF,
         typename IW, typename IGCC, typename IGCA,
         typename IPC, typename IRC, typename IT>
void
dbuct<INH, IC, IGVis, IGD, ISD, IBS, IFo, IBa, IGTF, IW, IGCC, IGCA, IPC, IRC, IT>::terminate_and_backtrack()
{
    terminate_.terminate();

    while (get_top_frame_.top().visit_lump >= get_top_frame_.top().budget)
        backstep_.backstep();

    in_rollout_ = false;
}

template<typename INH, typename IC, typename IGVis,
         typename IGD, typename ISD, typename IBS,
         typename IFo, typename IBa, typename IGTF,
         typename IW, typename IGCC, typename IGCA,
         typename IPC, typename IRC, typename IT>
bool
dbuct<INH, IC, IGVis, IGD, ISD, IBS, IFo, IBa, IGTF, IW, IGCC, IGCA, IPC, IRC, IT>::in_rollout() const
{
    return in_rollout_;
}

}

#endif
