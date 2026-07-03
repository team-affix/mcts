#ifndef DBUCT_HPP
#define DBUCT_HPP

#include <cmath>
#include <limits>
#include <optional>
#include <stack>

namespace monte_carlo
{

// Standard template parameter order:
//   1. domain types  — INodeHandle, IChoice, IFloat
//   2. read stats    — IGetVisits, IGetValue
//   3. write stats   — ISetVisits, ISetValue
//   4. graph         — IWalker
//   5. choice access — IGetChoiceCount, IGetChoiceAt
//   6. rollout       — IRolloutChoose
//
// Policy requirements (same as sim except IGetValueDelta is absent):
//   IGetVisits:      get_visits(const INodeHandle&) -> size_t   -- 0 if unseen
//   IGetValue:       get_value(const INodeHandle&)  -> IFloat   -- 0 if unseen
//   ISetVisits:      set_visits(const INodeHandle&, size_t) -> void
//   ISetValue:       set_value(const INodeHandle&, IFloat)  -> void
//   IWalker:         walk(const INodeHandle&, const IChoice&) -> INodeHandle
//   IGetChoiceCount: size() -> size_t
//   IGetChoiceAt:    at(size_t) -> IChoice
//   IRolloutChoose:  rollout_choose(const IGetChoiceCount&, const IGetChoiceAt&) -> IChoice
//
// Caller contract:
//   - Create one dbuct for the entire training run (not one per episode).
//   - Drive each episode: call choose() per step until terminal, then
//     call terminate(reward) which returns the node to resume from.
//   - Reset game state to the returned node before starting the next episode.
//   - Pass std::nullopt to terminate() if the episode ended without a valid
//     reward (e.g. the reached state is impossible). The budget slot is still
//     consumed but no stats are updated.
//
// Zero-default contract: IGetVisits and IGetValue must return 0 for unseen handles.
//
// Grant formula:
//   grant(N) = 1 + N / grant_increment_interval
//   k        = min(grant(N), remaining_budget)
//
//   grant_increment_interval = SIZE_MAX  ->  k = 1 always  ->  vanilla UCT.
//   grant_increment_interval must be > 0.
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
    typename IWalker,
    typename IGetChoiceCount,
    typename IGetChoiceAt,
    typename IRolloutChoose
>
struct dbuct
{
    dbuct(IGetVisits&     get_visits,
          IGetValue&      get_value,
          ISetVisits&     set_visits,
          ISetValue&      set_value,
          IWalker&        walker,
          IRolloutChoose& rollout,
          INodeHandle     root,
          size_t          grant_increment_interval,
          IFloat          exploration_constant);

    IChoice     choose(const IGetChoiceCount& get_choice_count,
                       const IGetChoiceAt&    get_choice_at);

    INodeHandle terminate(std::optional<IFloat> reward);

    size_t length() const;

private:
    struct frame
    {
        INodeHandle handle;
        size_t      remaining_budget;
        size_t      visit_lump; // subtree sim count accumulated; deposited to parent on pop
        IFloat      value_lump; // subtree reward sum accumulated; deposited to parent on pop
    };

    IGetVisits&     get_visits_;
    IGetValue&      get_value_;
    ISetVisits&     set_visits_;
    ISetValue&      set_value_;
    IWalker&        walker_;
    IRolloutChoose& rollout_;
    IFloat          exploration_constant_;
    size_t          grant_increment_interval_;

    std::stack<frame> stack_;
    bool              in_rollout_;

    void add_visits(size_t v);
    void add_value(IFloat l);
    void backtrack();
};

// ---------------------------------------------------------------------------
// member function definitions
// ---------------------------------------------------------------------------

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue,
         typename ISetVisits, typename ISetValue,
         typename IWalker,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IRolloutChoose>
dbuct<INodeHandle, IChoice, IFloat,
      IGetVisits, IGetValue, ISetVisits, ISetValue,
      IWalker,
      IGetChoiceCount, IGetChoiceAt,
      IRolloutChoose>::dbuct(
        IGetVisits&     get_visits,
        IGetValue&      get_value,
        ISetVisits&     set_visits,
        ISetValue&      set_value,
        IWalker&        walker,
        IRolloutChoose& rollout,
        INodeHandle     root,
        size_t          grant_increment_interval,
        IFloat          exploration_constant)
    : get_visits_(get_visits)
    , get_value_(get_value)
    , set_visits_(set_visits)
    , set_value_(set_value)
    , walker_(walker)
    , rollout_(rollout)
    , exploration_constant_(exploration_constant)
    , grant_increment_interval_(grant_increment_interval)
    , in_rollout_(false)
{
    stack_.push({root, std::numeric_limits<size_t>::max(), 0, IFloat{0}});
}

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue,
         typename ISetVisits, typename ISetValue,
         typename IWalker,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IRolloutChoose>
IChoice
dbuct<INodeHandle, IChoice, IFloat,
      IGetVisits, IGetValue, ISetVisits, ISetValue,
      IWalker,
      IGetChoiceCount, IGetChoiceAt,
      IRolloutChoose>::choose(
        const IGetChoiceCount& get_choice_count,
        const IGetChoiceAt&    get_choice_at)
{
    frame& current   = stack_.top();
    size_t current_visits = get_visits_.get_visits(current.handle);

    if (!in_rollout_ && current_visits == 0)
        in_rollout_ = true;

    if (in_rollout_)
        return rollout_.rollout_choose(get_choice_count, get_choice_at);

    // UCB1 selection.
    IFloat best_score = -std::numeric_limits<IFloat>::infinity();
    size_t best_i     = 0;
    size_t n          = get_choice_count.size();

    for (size_t i = 0; i < n; ++i)
    {
        IChoice           candidate = get_choice_at.at(i);
        const INodeHandle child     = walker_.walk(current.handle, candidate);
        size_t            child_v   = get_visits_.get_visits(child);

        if (child_v == 0)
        {
            best_score = std::numeric_limits<IFloat>::infinity();
            best_i     = i;
            break;
        }

        IFloat exploit = get_value_.get_value(child) / static_cast<IFloat>(child_v);
        IFloat explore = std::sqrt(
            std::log(static_cast<IFloat>(current_visits))
            / static_cast<IFloat>(child_v));
        IFloat score   = exploit + exploration_constant_ * explore;

        if (score > best_score)
        {
            best_score = score;
            best_i     = i;
        }
    }

    IChoice     chosen       = get_choice_at.at(best_i);
    INodeHandle child_handle = walker_.walk(current.handle, chosen);

    size_t grant_k = std::min(1 + current_visits / grant_increment_interval_,
                              current.remaining_budget);

    current.remaining_budget -= grant_k;
    stack_.push({child_handle, grant_k, 0, IFloat{0}});

    return chosen;
}

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue,
         typename ISetVisits, typename ISetValue,
         typename IWalker,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IRolloutChoose>
INodeHandle
dbuct<INodeHandle, IChoice, IFloat,
      IGetVisits, IGetValue, ISetVisits, ISetValue,
      IWalker,
      IGetChoiceCount, IGetChoiceAt,
      IRolloutChoose>::terminate(std::optional<IFloat> reward)
{
    in_rollout_ = false;

    frame& current = stack_.top();

    if (!reward.has_value())
    {
        if (current.remaining_budget != std::numeric_limits<size_t>::max())
            --current.remaining_budget;
        while (!stack_.empty() && stack_.top().remaining_budget == 0)
            stack_.pop();
        return stack_.top().handle;
    }

    if (current.remaining_budget != std::numeric_limits<size_t>::max())
        --current.remaining_budget;
    add_visits(1);
    add_value(*reward);

    while (!stack_.empty() && stack_.top().remaining_budget == 0)
        backtrack();

    return stack_.top().handle;
}

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue,
         typename ISetVisits, typename ISetValue,
         typename IWalker,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IRolloutChoose>
void
dbuct<INodeHandle, IChoice, IFloat,
      IGetVisits, IGetValue, ISetVisits, ISetValue,
      IWalker,
      IGetChoiceCount, IGetChoiceAt,
      IRolloutChoose>::add_visits(size_t v)
{
    frame& f = stack_.top();
    set_visits_.set_visits(f.handle, get_visits_.get_visits(f.handle) + v);
    f.visit_lump += v;
}

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue,
         typename ISetVisits, typename ISetValue,
         typename IWalker,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IRolloutChoose>
void
dbuct<INodeHandle, IChoice, IFloat,
      IGetVisits, IGetValue, ISetVisits, ISetValue,
      IWalker,
      IGetChoiceCount, IGetChoiceAt,
      IRolloutChoose>::add_value(IFloat l)
{
    frame& f = stack_.top();
    set_value_.set_value(f.handle, get_value_.get_value(f.handle) + l);
    f.value_lump += l;
}

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue,
         typename ISetVisits, typename ISetValue,
         typename IWalker,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IRolloutChoose>
void
dbuct<INodeHandle, IChoice, IFloat,
      IGetVisits, IGetValue, ISetVisits, ISetValue,
      IWalker,
      IGetChoiceCount, IGetChoiceAt,
      IRolloutChoose>::backtrack()
{
    size_t v = stack_.top().visit_lump;
    IFloat l = stack_.top().value_lump;
    stack_.pop();
    add_visits(v);
    add_value(l);
}

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue,
         typename ISetVisits, typename ISetValue,
         typename IWalker,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IRolloutChoose>
size_t
dbuct<INodeHandle, IChoice, IFloat,
      IGetVisits, IGetValue, ISetVisits, ISetValue,
      IWalker,
      IGetChoiceCount, IGetChoiceAt,
      IRolloutChoose>::length() const
{
    return stack_.size() - 1;
}

} // namespace monte_carlo

#endif // DBUCT_HPP
