#ifndef DBUCT_HPP
#define DBUCT_HPP

#include <cmath>
#include <limits>
#include <stack>

namespace monte_carlo
{

// Standard template parameter order:
//   1. domain types    — INodeHandle, IChoice, IFloat
//   2. read stats      — IGetVisits, IGetValue
//   3. write stats     — ISetVisits, ISetValue
//   4. dispatch stats  — IGetDispatches, ISetDispatches
//   5. batch policy    — IComputeBatchSize
//   6. graph           — IWalker
//   7. choice access   — IGetChoiceCount, IGetChoiceAt
//   8. rollout         — IRolloutChoose
//
// Policy requirements:
//   IGetVisits:       get_visits(const INodeHandle&) -> size_t   -- 0 if unseen
//   IGetValue:        get_value(const INodeHandle&)  -> IFloat   -- 0 if unseen
//   ISetVisits:       set_visits(const INodeHandle&, size_t) -> void
//   ISetValue:        set_value(const INodeHandle&, IFloat)  -> void
//   IGetDispatches:   get_dispatches(const INodeHandle&) -> size_t  -- 0 if unseen
//   ISetDispatches:   set_dispatches(const INodeHandle&, size_t) -> void
//   IComputeBatchSize:compute_batch_size(size_t dispatch_count) -> size_t
//   IWalker:          walk(const INodeHandle&, const IChoice&) -> INodeHandle
//   IGetChoiceCount:  size() -> size_t
//   IGetChoiceAt:     at(size_t) -> IChoice
//   IRolloutChoose:   rollout_choose(const IGetChoiceCount&, const IGetChoiceAt&) -> IChoice
//
// Caller contract:
//   - Create one dbuct for the entire training run (not one per episode).
//   - Maintain your own path stack of visited nodes alongside the episode.
//   - Drive each episode: call choose() per step until terminal, then call
//     terminate(reward).  Sync your path via path.resize(depth()).
//   - Optionally call backstep() after terminate() to climb toward root.
//     Every ancestor of the resume node has nonzero budget remaining, so
//     backstep() is always safe before the next sim.
//
// Budget invariant: for each node n, budget(parent(n)) >= budget(n).
// terminate() backs up until the first frame with nonzero remaining budget.
//
// Zero-default contract: IGetVisits, IGetValue, IGetDispatches must return 0
//   for unseen handles.
//
// Grant formula:
//   IComputeBatchSize::compute_batch_size(D) is called with the pre-increment
//   dispatch count D for the current node.  linear_batch_increment implements
//   1 + D / B, where B is the grant_increment_interval.
//   B = SIZE_MAX gives compute_batch_size = 1 always, equivalent to vanilla UCT.
//
// UCB1:
//   exploit = get_value(child) / get_visits(child)
//   explore = c * sqrt( ln(get_visits(parent)) / get_visits(child) )

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
    typename IRolloutChoose
>
struct dbuct
{
    dbuct(IGetVisits&      get_visits,
          IGetValue&       get_value,
          ISetVisits&      set_visits,
          ISetValue&       set_value,
          IGetDispatches&  get_dispatches,
          ISetDispatches&  set_dispatches,
          IComputeBatchSize& compute_batch_size,
          IWalker&         walker,
          IRolloutChoose&  rollout,
          INodeHandle      root,
          IFloat           exploration_constant);

    IChoice choose(const IGetChoiceCount& get_choice_count,
                   const IGetChoiceAt&    get_choice_at);

    // Backpropagate the terminal reward, then backstep until the top frame
    // has nonzero remaining budget (visit_lump < budget).
    void terminate(IFloat reward);

    // Pop one frame and deposit its lump into the parent.  Caller-controlled
    // backtracking after terminate(); safe while depth() > 0.
    void backstep();

    size_t depth() const { return stack_.size(); }
    bool   in_rollout() const { return in_rollout_; }

private:
    struct frame
    {
        INodeHandle handle;
        size_t      budget;
        size_t      visit_lump; // subtree sim count accumulated; deposited to parent on pop
        IFloat      value_lump; // subtree reward sum accumulated; deposited to parent on pop
    };

    IGetVisits&       get_visits_;
    IGetValue&        get_value_;
    ISetVisits&       set_visits_;
    ISetValue&        set_value_;
    IGetDispatches&   get_dispatches_;
    ISetDispatches&   set_dispatches_;
    IComputeBatchSize& compute_batch_size_;
    IWalker&          walker_;
    IRolloutChoose&   rollout_;
    IFloat            exploration_constant_;

    std::stack<frame> stack_;
    bool              in_rollout_;

    void add_visits(size_t v);
    void add_value(IFloat l);
};

// ---------------------------------------------------------------------------
// member function definitions
// ---------------------------------------------------------------------------

// Abbreviations used in template heads below:
//   INH  = INodeHandle      IC   = IChoice         IF   = IFloat
//   IGVis= IGetVisits        IGVal= IGetValue
//   ISVis= ISetVisits        ISVal= ISetValue
//   IGD  = IGetDispatches   ISD  = ISetDispatches   IBS  = IComputeBatchSize
//   IW   = IWalker          IGCC = IGetChoiceCount  IGCA = IGetChoiceAt
//   IRC  = IRolloutChoose

template<typename INH, typename IC, typename IF,
         typename IGVis, typename IGVal, typename ISVis, typename ISVal,
         typename IGD, typename ISD, typename IBS,
         typename IW, typename IGCC, typename IGCA, typename IRC>
dbuct<INH, IC, IF, IGVis, IGVal, ISVis, ISVal, IGD, ISD, IBS, IW, IGCC, IGCA, IRC>::dbuct(
        IGVis& get_visits,
        IGVal& get_value,
        ISVis& set_visits,
        ISVal& set_value,
        IGD&   get_dispatches,
        ISD&   set_dispatches,
        IBS&   compute_batch_size,
        IW&    walker,
        IRC&   rollout,
        INH    root,
        IF     exploration_constant)
    : get_visits_(get_visits)
    , get_value_(get_value)
    , set_visits_(set_visits)
    , set_value_(set_value)
    , get_dispatches_(get_dispatches)
    , set_dispatches_(set_dispatches)
    , compute_batch_size_(compute_batch_size)
    , walker_(walker)
    , rollout_(rollout)
    , exploration_constant_(exploration_constant)
    , in_rollout_(false)
{
    stack_.push({root, std::numeric_limits<size_t>::max(), 0, IF{0}});
}

template<typename INH, typename IC, typename IF,
         typename IGVis, typename IGVal, typename ISVis, typename ISVal,
         typename IGD, typename ISD, typename IBS,
         typename IW, typename IGCC, typename IGCA, typename IRC>
IC
dbuct<INH, IC, IF, IGVis, IGVal, ISVis, ISVal, IGD, ISD, IBS, IW, IGCC, IGCA, IRC>::choose(
        const IGCC& get_choice_count,
        const IGCA& get_choice_at)
{
    frame& current        = stack_.top();
    size_t current_visits = get_visits_.get_visits(current.handle);

    if (!in_rollout_ && current_visits == 0)
        in_rollout_ = true;

    if (in_rollout_)
        return rollout_.rollout_choose(get_choice_count, get_choice_at);

    // UCB1 selection.
    IF     best_score = -std::numeric_limits<IF>::infinity();
    size_t best_i     = 0;
    size_t n          = get_choice_count.size();

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
        IF explore = std::sqrt(
            std::log(static_cast<IF>(current_visits))
            / static_cast<IF>(child_v));
        IF score   = exploit + exploration_constant_ * explore;

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

    return chosen;
}

template<typename INH, typename IC, typename IF,
         typename IGVis, typename IGVal, typename ISVis, typename ISVal,
         typename IGD, typename ISD, typename IBS,
         typename IW, typename IGCC, typename IGCA, typename IRC>
void
dbuct<INH, IC, IF, IGVis, IGVal, ISVis, ISVal, IGD, ISD, IBS, IW, IGCC, IGCA, IRC>::terminate(
        IF reward)
{
    add_visits(1);
    add_value(reward);

    while (stack_.top().visit_lump >= stack_.top().budget)
        backstep();

    in_rollout_ = false;
}

template<typename INH, typename IC, typename IF,
         typename IGVis, typename IGVal, typename ISVis, typename ISVal,
         typename IGD, typename ISD, typename IBS,
         typename IW, typename IGCC, typename IGCA, typename IRC>
void
dbuct<INH, IC, IF, IGVis, IGVal, ISVis, ISVal, IGD, ISD, IBS, IW, IGCC, IGCA, IRC>::add_visits(
        size_t v)
{
    frame& f = stack_.top();
    set_visits_.set_visits(f.handle, get_visits_.get_visits(f.handle) + v);
    f.visit_lump += v;
}

template<typename INH, typename IC, typename IF,
         typename IGVis, typename IGVal, typename ISVis, typename ISVal,
         typename IGD, typename ISD, typename IBS,
         typename IW, typename IGCC, typename IGCA, typename IRC>
void
dbuct<INH, IC, IF, IGVis, IGVal, ISVis, ISVal, IGD, ISD, IBS, IW, IGCC, IGCA, IRC>::add_value(
        IF l)
{
    frame& f = stack_.top();
    set_value_.set_value(f.handle, get_value_.get_value(f.handle) + l);
    f.value_lump += l;
}

template<typename INH, typename IC, typename IF,
         typename IGVis, typename IGVal, typename ISVis, typename ISVal,
         typename IGD, typename ISD, typename IBS,
         typename IW, typename IGCC, typename IGCA, typename IRC>
void
dbuct<INH, IC, IF, IGVis, IGVal, ISVis, ISVal, IGD, ISD, IBS, IW, IGCC, IGCA, IRC>::backstep()
{
    const frame& current = stack_.top();
    size_t v = current.visit_lump;
    IF     l = current.value_lump;
    stack_.pop();
    add_visits(v);
    add_value(l);
}

} // namespace monte_carlo

#endif // DBUCT_HPP
