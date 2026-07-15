#ifndef SIM_HPP
#define SIM_HPP

#include <cmath>
#include <limits>
#include <vector>

namespace monte_carlo
{

// Standard template parameter order:
//   1. domain types  — INodeHandle, IChoice, IFloat
//   2. read stats    — IGetVisits, IGetValue
//   3. write stats   — ISetVisits, ISetValue
//   4. graph         — IWalker
//   5. choice access — IGetChoiceCount, IGetChoiceAt
//   6. rollout       — IRolloutChoose
//   7. value delta   — IGetValueDelta
//
// Policy requirements:
//   IGetVisits:      get_visits(const INodeHandle&) -> size_t   -- 0 if unseen
//   IGetValue:       get_value(const INodeHandle&)  -> IFloat   -- 0 if unseen
//   ISetVisits:      set_visits(const INodeHandle&, size_t) -> void
//   ISetValue:       set_value(const INodeHandle&, IFloat)  -> void
//   IWalker:         walk(const INodeHandle&, const IChoice&) -> INodeHandle
//   IGetChoiceCount: size() -> size_t
//   IGetChoiceAt:    at(size_t) -> IChoice
//   IRolloutChoose:  rollout_choose(const IGetChoiceCount&, const IGetChoiceAt&) -> IChoice
//   IGetValueDelta:  get_value_delta(const INodeHandle&) -> IFloat
//                      -- called per path-node during terminate()
//
// Caller contract:
//   - Drive the game loop; call choose() for each step until the terminal state.
//   - Call terminate() once the episode ends. It backpropagates the per-node
//     delta (from IGetValueDelta) to every node on the selection path.
//
// Zero-default contract: IGetVisits and IGetValue must return 0 for unseen handles.
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
    typename IRolloutChoose,
    typename IGetValueDelta
>
struct sim
{
    sim(IGetVisits&        get_visits,
        IGetValue&         get_value,
        ISetVisits&        set_visits,
        ISetValue&         set_value,
        IWalker&           walker,
        IRolloutChoose&    rollout,
        IGetValueDelta&    value_delta,
        INodeHandle root,
        IFloat             exploration_constant);

    IChoice choose(const IGetChoiceCount& get_choice_count, const IGetChoiceAt& get_choice_at);
    void    terminate();
    size_t  length() const;

private:
    IGetVisits&     get_visits_;
    IGetValue&      get_value_;
    ISetVisits&     set_visits_;
    ISetValue&      set_value_;
    IWalker&        walker_;
    IRolloutChoose& rollout_;
    IGetValueDelta& value_delta_;
    IFloat          exploration_constant_;

    INodeHandle              current_node_;
    std::vector<INodeHandle> backprop_path_;
    size_t                   sim_length_;
    bool                     in_rollout_;
};

// ---------------------------------------------------------------------------
// member function definitions
// ---------------------------------------------------------------------------

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue,
         typename ISetVisits, typename ISetValue,
         typename IWalker,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IRolloutChoose,
         typename IGetValueDelta>
sim<INodeHandle, IChoice, IFloat,
    IGetVisits, IGetValue, ISetVisits, ISetValue,
    IWalker,
    IGetChoiceCount, IGetChoiceAt,
    IRolloutChoose,
    IGetValueDelta>::sim(
        IGetVisits&        get_visits,
        IGetValue&         get_value,
        ISetVisits&        set_visits,
        ISetValue&         set_value,
        IWalker&           walker,
        IRolloutChoose&    rollout,
        IGetValueDelta&    value_delta,
        INodeHandle root,
        IFloat             exploration_constant)
    : get_visits_(get_visits)
    , get_value_(get_value)
    , set_visits_(set_visits)
    , set_value_(set_value)
    , walker_(walker)
    , rollout_(rollout)
    , value_delta_(value_delta)
    , exploration_constant_(exploration_constant)
    , current_node_(root)
    , backprop_path_({root})
    , sim_length_(0)
    , in_rollout_(false)
{}

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue,
         typename ISetVisits, typename ISetValue,
         typename IWalker,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IRolloutChoose,
         typename IGetValueDelta>
IChoice
sim<INodeHandle, IChoice, IFloat,
    IGetVisits, IGetValue, ISetVisits, ISetValue,
    IWalker,
    IGetChoiceCount, IGetChoiceAt,
    IRolloutChoose,
    IGetValueDelta>::choose(
        const IGetChoiceCount& get_choice_count,
        const IGetChoiceAt&    get_choice_at)
{
    ++sim_length_;

    if (in_rollout_)
    {
        IChoice chosen = rollout_.rollout_choose(get_choice_count, get_choice_at);
        current_node_  = walker_.walk(current_node_, chosen);
        return chosen;
    }

    // UCB1 selection.
    IFloat best_score = -std::numeric_limits<IFloat>::infinity();
    size_t best_i     = 0;
    size_t n          = get_choice_count.size();

    for (size_t i = 0; i < n; ++i)
    {
        IChoice           candidate  = get_choice_at.at(i);
        const INodeHandle child_node = walker_.walk(current_node_, candidate);
        size_t            child_v    = get_visits_.get_visits(child_node);

        if (child_v == 0)
        {
            best_score = std::numeric_limits<IFloat>::infinity();
            best_i     = i;
            break;
        }

        IFloat exploit = get_value_.get_value(child_node) / static_cast<IFloat>(child_v);
        IFloat explore = std::sqrt(
            std::log(static_cast<IFloat>(get_visits_.get_visits(current_node_)))
            / static_cast<IFloat>(child_v));
        IFloat score   = exploit + exploration_constant_ * explore;

        if (score > best_score)
        {
            best_score = score;
            best_i     = i;
        }
    }

    IChoice     chosen       = get_choice_at.at(best_i);
    INodeHandle chosen_child = walker_.walk(current_node_, chosen);
    backprop_path_.push_back(chosen_child);
    current_node_ = chosen_child;

    size_t child_visits = get_visits_.get_visits(chosen_child);
    
    if (child_visits == 0)
        in_rollout_ = true;

    return chosen;
}

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue,
         typename ISetVisits, typename ISetValue,
         typename IWalker,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IRolloutChoose,
         typename IGetValueDelta>
void
sim<INodeHandle, IChoice, IFloat,
    IGetVisits, IGetValue, ISetVisits, ISetValue,
    IWalker,
    IGetChoiceCount, IGetChoiceAt,
    IRolloutChoose,
    IGetValueDelta>::terminate()
{
    for (const INodeHandle& node : backprop_path_)
    {
        set_visits_.set_visits(node, get_visits_.get_visits(node) + 1);
        set_value_.set_value(node,   get_value_.get_value(node)
                                     + value_delta_.get_value_delta(node));
    }
}

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue,
         typename ISetVisits, typename ISetValue,
         typename IWalker,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IRolloutChoose,
         typename IGetValueDelta>
size_t
sim<INodeHandle, IChoice, IFloat,
    IGetVisits, IGetValue, ISetVisits, ISetValue,
    IWalker,
    IGetChoiceCount, IGetChoiceAt,
    IRolloutChoose,
    IGetValueDelta>::length() const
{
    return sim_length_;
}

} // namespace monte_carlo

#endif // SIM_HPP
