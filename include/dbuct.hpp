#ifndef DBUCT_HPP
#define DBUCT_HPP

#include <cmath>
#include <limits>
#include <stack>

namespace monte_carlo
{

template<
    typename INodeHandle,
    typename IChoice,
    typename IFloat,
    typename IGetVisits,
    typename IGetValue,
    typename ISetVisits,
    typename ISetValue,
    typename IGetDispatches,
    typename ISetDispatches,
    typename IComputeBatchSize,
    typename IWalker,
    typename IGetChoiceCount,
    typename IGetChoiceAt,
    typename IRolloutChoose,
    typename IGetValueDelta,
    typename IGetExplorationConstant
>
struct dbuct
{
    dbuct(IGetVisits&              get_visits,
          IGetValue&               get_value,
          ISetVisits&              set_visits,
          ISetValue&               set_value,
          IGetDispatches&          get_dispatches,
          ISetDispatches&          set_dispatches,
          IComputeBatchSize&       compute_batch_size,
          IWalker&                 walker,
          IRolloutChoose&          rollout,
          IGetValueDelta&          value_delta,
          IGetExplorationConstant& get_exploration_constant,
          INodeHandle              root);

    IChoice choose(const IGetChoiceCount& get_choice_count,
                   const IGetChoiceAt&    get_choice_at);

    void terminate();

    void backstep();

    size_t depth() const { return stack_.size(); }
    bool   in_rollout() const { return in_rollout_; }

private:
    struct frame
    {
        INodeHandle handle;
        size_t      budget;
        size_t      visit_lump;
        IFloat      value_lump;
    };

    IGetVisits&              get_visits_;
    IGetValue&               get_value_;
    ISetVisits&              set_visits_;
    ISetValue&               set_value_;
    IGetDispatches&          get_dispatches_;
    ISetDispatches&          set_dispatches_;
    IComputeBatchSize&       compute_batch_size_;
    IWalker&                 walker_;
    IRolloutChoose&          rollout_;
    IGetValueDelta&          value_delta_;
    IGetExplorationConstant& get_exploration_constant_;

    std::stack<frame> stack_;
    bool              in_rollout_;

    void add_visits(size_t v);
    void add_value(IFloat l);
};

// Legend: INH=INodeHandle, IC=IChoice, IF=IFloat, IGVis=IGetVisits, IGVal=IGetValue,
//         ISVis=ISetVisits, ISVal=ISetValue, IGD=IGetDispatches, ISD=ISetDispatches,
//         IBS=IComputeBatchSize, IW=IWalker, IGCC=IGetChoiceCount, IGCA=IGetChoiceAt,
//         IRC=IRolloutChoose, IGVD=IGetValueDelta, IGEC=IGetExplorationConstant

template<typename INH, typename IC, typename IF,
         typename IGVis, typename IGVal, typename ISVis, typename ISVal,
         typename IGD, typename ISD, typename IBS,
         typename IW, typename IGCC, typename IGCA, typename IRC, typename IGVD, typename IGEC>
dbuct<INH, IC, IF, IGVis, IGVal, ISVis, ISVal, IGD, ISD, IBS, IW, IGCC, IGCA, IRC, IGVD, IGEC>::dbuct(
        IGVis& get_visits,
        IGVal& get_value,
        ISVis& set_visits,
        ISVal& set_value,
        IGD&   get_dispatches,
        ISD&   set_dispatches,
        IBS&   compute_batch_size,
        IW&    walker,
        IRC&   rollout,
        IGVD&  value_delta,
        IGEC&  get_exploration_constant,
        INH    root)
    : get_visits_(get_visits)
    , get_value_(get_value)
    , set_visits_(set_visits)
    , set_value_(set_value)
    , get_dispatches_(get_dispatches)
    , set_dispatches_(set_dispatches)
    , compute_batch_size_(compute_batch_size)
    , walker_(walker)
    , rollout_(rollout)
    , value_delta_(value_delta)
    , get_exploration_constant_(get_exploration_constant)
    , in_rollout_(false)
{
    stack_.push({root, std::numeric_limits<size_t>::max(), 0, IF{0}});
}

template<typename INH, typename IC, typename IF,
         typename IGVis, typename IGVal, typename ISVis, typename ISVal,
         typename IGD, typename ISD, typename IBS,
         typename IW, typename IGCC, typename IGCA, typename IRC, typename IGVD, typename IGEC>
IC
dbuct<INH, IC, IF, IGVis, IGVal, ISVis, ISVal, IGD, ISD, IBS, IW, IGCC, IGCA, IRC, IGVD, IGEC>::choose(
        const IGCC& get_choice_count,
        const IGCA& get_choice_at)
{
    frame& current        = stack_.top();
    size_t current_visits = get_visits_.get_visits(current.handle);
    
    if (in_rollout_)
        return rollout_.rollout_choose(get_choice_count, get_choice_at);

    IF     best_score = -std::numeric_limits<IF>::infinity();
    size_t best_i     = 0;
    size_t n          = get_choice_count.size();
    IF     c          = get_exploration_constant_.get_exploration_constant(current.handle);
    IF     ln_parent  = std::log(static_cast<IF>(current_visits));

    for (size_t i = 0; i < n; ++i)
    {
        IC        candidate = get_choice_at.at(i);
        const INH child     = walker_.walk(current.handle, candidate);
        size_t    child_v   = get_visits_.get_visits(child);

        if (child_v == 0)
        {
            best_score = std::numeric_limits<IF>::infinity();
            best_i     = i;
            break;
        }

        IF exploit = get_value_.get_value(child) / static_cast<IF>(child_v);
        IF explore = std::sqrt(ln_parent / static_cast<IF>(child_v));
        IF score   = exploit + c * explore;

        if (score > best_score)
        {
            best_score = score;
            best_i     = i;
        }
    }

    IC  chosen       = get_choice_at.at(best_i);
    INH child_handle = walker_.walk(current.handle, chosen);

    size_t current_dispatches = get_dispatches_.get_dispatches(current.handle);
    size_t remaining_budget   = current.budget - current.visit_lump;
    size_t grant_k = std::min(
        compute_batch_size_.compute_batch_size(current_dispatches),
        remaining_budget);
    set_dispatches_.set_dispatches(current.handle, current_dispatches + 1);

    stack_.push({child_handle, grant_k, 0, IF{0}});

    size_t child_visits = get_visits_.get_visits(child_handle);
    
    // expansion+rollout phase (frame already pushed so expansion done)
    if (child_visits == 0)
        in_rollout_ = true;

    return chosen;
}

template<typename INH, typename IC, typename IF,
         typename IGVis, typename IGVal, typename ISVis, typename ISVal,
         typename IGD, typename ISD, typename IBS,
         typename IW, typename IGCC, typename IGCA, typename IRC, typename IGVD, typename IGEC>
void
dbuct<INH, IC, IF, IGVis, IGVal, ISVis, ISVal, IGD, ISD, IBS, IW, IGCC, IGCA, IRC, IGVD, IGEC>::terminate()
{
    add_visits(1);
    add_value(value_delta_.get_value_delta(stack_.top().handle));

    while (stack_.top().visit_lump >= stack_.top().budget)
        backstep();

    in_rollout_ = false;
}

template<typename INH, typename IC, typename IF,
         typename IGVis, typename IGVal, typename ISVis, typename ISVal,
         typename IGD, typename ISD, typename IBS,
         typename IW, typename IGCC, typename IGCA, typename IRC, typename IGVD, typename IGEC>
void
dbuct<INH, IC, IF, IGVis, IGVal, ISVis, ISVal, IGD, ISD, IBS, IW, IGCC, IGCA, IRC, IGVD, IGEC>::add_visits(
        size_t v)
{
    frame& f = stack_.top();
    set_visits_.set_visits(f.handle, get_visits_.get_visits(f.handle) + v);
    f.visit_lump += v;
}

template<typename INH, typename IC, typename IF,
         typename IGVis, typename IGVal, typename ISVis, typename ISVal,
         typename IGD, typename ISD, typename IBS,
         typename IW, typename IGCC, typename IGCA, typename IRC, typename IGVD, typename IGEC>
void
dbuct<INH, IC, IF, IGVis, IGVal, ISVis, ISVal, IGD, ISD, IBS, IW, IGCC, IGCA, IRC, IGVD, IGEC>::add_value(
        IF l)
{
    frame& f = stack_.top();
    set_value_.set_value(f.handle, get_value_.get_value(f.handle) + l);
    f.value_lump += l;
}

template<typename INH, typename IC, typename IF,
         typename IGVis, typename IGVal, typename ISVis, typename ISVal,
         typename IGD, typename ISD, typename IBS,
         typename IW, typename IGCC, typename IGCA, typename IRC, typename IGVD, typename IGEC>
void
dbuct<INH, IC, IF, IGVis, IGVal, ISVis, ISVal, IGD, ISD, IBS, IW, IGCC, IGCA, IRC, IGVD, IGEC>::backstep()
{
    const frame& current = stack_.top();
    size_t v = current.visit_lump;
    IF     l = current.value_lump;
    stack_.pop();
    add_visits(v);
    add_value(l);
}

}

#endif
