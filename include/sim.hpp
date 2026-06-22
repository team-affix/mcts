#ifndef SIM_HPP
#define SIM_HPP

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace monte_carlo
{

// Standard template parameter order:
//   1. domain types   — INodeHandle, IChoice, IFloat
//   2. read stats     — IGetVisits, IGetValue
//   3. write stats    — ISetVisits, ISetValue
//   4. graph/reward   — IWalker, IGetArrivalReward
//   5. edge tracking  — IGetEdgeVisits, ISetEdgeVisits
//   6. choice access  — IGetChoiceCount, IGetChoiceAt
//   7. rollout        — IRolloutChoose
//
// Policy requirements:
//   IGetVisits:       get_visits(const INodeHandle&) -> size_t       -- 0 if node unseen
//   IGetValue:        get_value(const INodeHandle&)  -> IFloat        -- 0 if node unseen
//   ISetVisits:       set_visits(const INodeHandle&, size_t) -> void
//   ISetValue:        set_value(const INodeHandle&, IFloat)  -> void
//   IWalker:          walker.walk(const INodeHandle&, IChoice) -> INodeHandle
//   IGetArrivalReward:get_arrival_reward(const INodeHandle& parent, IChoice) -> IFloat
//   IGetEdgeVisits:   get_edge_visits(const INodeHandle& parent, const INodeHandle& child) -> size_t
//   ISetEdgeVisits:   set_edge_visits(const INodeHandle& parent, const INodeHandle& child, size_t) -> void
//   IGetChoiceCount:  get_choice_count.size() -> size_t
//   IGetChoiceAt:     get_choice_at.at(size_t) -> IChoice
//   IRolloutChoose:   rollout.rollout_choose(IGetChoiceCount&, IGetChoiceAt&) -> IChoice
//
// Caller contract:
//   - After each choose(), call step_reward(r) with the immediate reward earned that step.
//   - When the game reaches a terminal state (e.g. out-of-bounds), call step_terminal()
//     BEFORE terminate().  This registers the terminal node in the bank so the explore term
//     for edges leading to it becomes finite on all subsequent visits.
//   - Call terminate() when the episode ends; it backpropagates accumulated rewards
//     to every node on the selection path and updates edge visit counts.
//
// UCB1 exploit + explore:
//   exploit = arrival_reward(parent, choice) + get_value(child) / get_visits(child)
//   explore = c * sqrt( ln(get_visits(parent)) / get_edge_visits(parent, child) )
//
//   The explore term uses per-edge visit counts, not global node visit counts.
//   This prevents nodes that are popular through other parents from appearing
//   "already explored" from a parent that has never actually chosen them.
//
// Zero-default contract: IGetVisits, IGetValue, IGetEdgeVisits must return 0 for
// any handle / pair not yet written.  sim uses 0 to detect unvisited nodes/edges.

template<
    typename INodeHandle,
    typename IChoice,
    typename IFloat,
    typename IGetVisits,
    typename IGetValue,
    typename ISetVisits,
    typename ISetValue,
    typename IWalker,
    typename IGetArrivalReward,
    typename IGetEdgeVisits,
    typename ISetEdgeVisits,
    typename IGetChoiceCount,
    typename IGetChoiceAt,
    typename IRolloutChoose
>
struct sim
{
    sim(IGetVisits&          get_visits,
        IGetValue&           get_value,
        ISetVisits&          set_visits,
        ISetValue&           set_value,
        IWalker&             walker,
        IGetArrivalReward&   arrival_reward,
        IGetEdgeVisits&      get_edge_visits,
        ISetEdgeVisits&      set_edge_visits,
        IRolloutChoose&      rollout,
        const INodeHandle&   root,
        IFloat               exploration_constant);

    IChoice choose(IGetChoiceCount& get_choice_count, IGetChoiceAt& get_choice_at);
    void    step_reward(IFloat r);
    void    step_terminal();
    void    terminate();
    size_t  length() const;

private:
    IGetVisits&         get_visits_;
    IGetValue&          get_value_;
    ISetVisits&         set_visits_;
    ISetValue&          set_value_;
    IWalker&            walker_;
    IGetArrivalReward&  arrival_reward_;
    IGetEdgeVisits&     get_edge_visits_;
    ISetEdgeVisits&     set_edge_visits_;
    IRolloutChoose&     rollout_;
    INodeHandle         root_;
    IFloat              exploration_constant_;

    INodeHandle                                    current_node_;
    std::vector<std::pair<INodeHandle, IFloat>>    backprop_path_;
    size_t                                         sim_length_;
    bool                                           in_rollout_;
};

// ---------------------------------------------------------------------------
// member function definitions
// ---------------------------------------------------------------------------

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue,
         typename ISetVisits, typename ISetValue,
         typename IWalker, typename IGetArrivalReward,
         typename IGetEdgeVisits, typename ISetEdgeVisits,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IRolloutChoose>
sim<INodeHandle, IChoice, IFloat,
    IGetVisits, IGetValue, ISetVisits, ISetValue,
    IWalker, IGetArrivalReward,
    IGetEdgeVisits, ISetEdgeVisits,
    IGetChoiceCount, IGetChoiceAt,
    IRolloutChoose>::sim(
        IGetVisits&        get_visits,
        IGetValue&         get_value,
        ISetVisits&        set_visits,
        ISetValue&         set_value,
        IWalker&           walker,
        IGetArrivalReward& arrival_reward,
        IGetEdgeVisits&    get_edge_visits,
        ISetEdgeVisits&    set_edge_visits,
        IRolloutChoose&    rollout,
        const INodeHandle& root,
        IFloat             exploration_constant)
    : get_visits_(get_visits)
    , get_value_(get_value)
    , set_visits_(set_visits)
    , set_value_(set_value)
    , walker_(walker)
    , arrival_reward_(arrival_reward)
    , get_edge_visits_(get_edge_visits)
    , set_edge_visits_(set_edge_visits)
    , rollout_(rollout)
    , root_(root)
    , exploration_constant_(exploration_constant)
    , current_node_(root)
    , backprop_path_{}
    , sim_length_(0)
    , in_rollout_(false)
{}

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue,
         typename ISetVisits, typename ISetValue,
         typename IWalker, typename IGetArrivalReward,
         typename IGetEdgeVisits, typename ISetEdgeVisits,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IRolloutChoose>
IChoice
sim<INodeHandle, IChoice, IFloat,
    IGetVisits, IGetValue, ISetVisits, ISetValue,
    IWalker, IGetArrivalReward,
    IGetEdgeVisits, ISetEdgeVisits,
    IGetChoiceCount, IGetChoiceAt,
    IRolloutChoose>::choose(
        IGetChoiceCount& get_choice_count,
        IGetChoiceAt&    get_choice_at)
{
    ++sim_length_;

    IChoice chosen;

    if (in_rollout_)
    {
        chosen = rollout_.rollout_choose(get_choice_count, get_choice_at);
    }
    else
    {
        backprop_path_.push_back({current_node_, IFloat{}});

        if (get_visits_.get_visits(current_node_) == 0)
        {
            in_rollout_ = true;
            chosen = rollout_.rollout_choose(get_choice_count, get_choice_at);
        }
        else
        {
            // UCB1 selection.
            // Exploit = arrival_reward(parent→child) + avg_future_value(child)
            // Explore = c * sqrt( ln(parent.visits) / edge_visits(parent, child) )
            //
            // Using per-edge visit counts in the explore term prevents a child that is
            // globally popular (visited many times via other parents) from being treated
            // as already explored from this specific parent.
            IFloat best_score = -std::numeric_limits<IFloat>::infinity();
            size_t best_i     = 0;
            size_t n          = get_choice_count.size();

            for (size_t i = 0; i < n; ++i)
            {
                IChoice           candidate  = get_choice_at.at(i);
                const INodeHandle child_node = walker_.walk(current_node_, candidate);
                size_t            edge_v     = get_edge_visits_.get_edge_visits(current_node_, child_node);

                IFloat score;
                if (edge_v == 0)
                {
                    score = std::numeric_limits<IFloat>::infinity();
                }
                else
                {
                    IFloat arrival = arrival_reward_.get_arrival_reward(current_node_, candidate);
                    IFloat exploit = arrival + get_value_.get_value(child_node)
                                              / static_cast<IFloat>(get_visits_.get_visits(child_node));
                    IFloat explore = std::sqrt(
                        std::log(static_cast<IFloat>(get_visits_.get_visits(current_node_)))
                        / static_cast<IFloat>(edge_v));
                    score = exploit + exploration_constant_ * explore;
                }

                if (score > best_score)
                {
                    best_score = score;
                    best_i     = i;
                }
            }

            chosen = get_choice_at.at(best_i);
        }
    }

    current_node_ = walker_.walk(current_node_, chosen);
    return chosen;
}

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue,
         typename ISetVisits, typename ISetValue,
         typename IWalker, typename IGetArrivalReward,
         typename IGetEdgeVisits, typename ISetEdgeVisits,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IRolloutChoose>
void
sim<INodeHandle, IChoice, IFloat,
    IGetVisits, IGetValue, ISetVisits, ISetValue,
    IWalker, IGetArrivalReward,
    IGetEdgeVisits, ISetEdgeVisits,
    IGetChoiceCount, IGetChoiceAt,
    IRolloutChoose>::step_reward(IFloat r)
{
    for (auto& [node_handle, future_reward] : backprop_path_)
        future_reward += r;
}

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue,
         typename ISetVisits, typename ISetValue,
         typename IWalker, typename IGetArrivalReward,
         typename IGetEdgeVisits, typename ISetEdgeVisits,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IRolloutChoose>
void
sim<INodeHandle, IChoice, IFloat,
    IGetVisits, IGetValue, ISetVisits, ISetValue,
    IWalker, IGetArrivalReward,
    IGetEdgeVisits, ISetEdgeVisits,
    IGetChoiceCount, IGetChoiceAt,
    IRolloutChoose>::step_terminal()
{
    // Push the terminal position so that the edge leading to it gets tracked
    // in terminate().  Terminal nodes always carry zero accumulated future reward.
    backprop_path_.push_back({current_node_, IFloat{}});
}

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue,
         typename ISetVisits, typename ISetValue,
         typename IWalker, typename IGetArrivalReward,
         typename IGetEdgeVisits, typename ISetEdgeVisits,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IRolloutChoose>
void
sim<INodeHandle, IChoice, IFloat,
    IGetVisits, IGetValue, ISetVisits, ISetValue,
    IWalker, IGetArrivalReward,
    IGetEdgeVisits, ISetEdgeVisits,
    IGetChoiceCount, IGetChoiceAt,
    IRolloutChoose>::terminate()
{
    // Update per-node visit counts and accumulated future rewards.
    for (const auto& [node_handle, future_reward] : backprop_path_)
    {
        set_visits_.set_visits(node_handle, get_visits_.get_visits(node_handle) + 1);
        set_value_.set_value(node_handle, get_value_.get_value(node_handle) + future_reward);
    }

    // Update per-edge visit counts from consecutive pairs on the selection path.
    if (backprop_path_.size() >= 2)
    {
        auto it      = backprop_path_.begin();
        auto next_it = std::next(it);
        while (next_it != backprop_path_.end())
        {
            const INodeHandle& parent = it->first;
            const INodeHandle& child  = next_it->first;
            set_edge_visits_.set_edge_visits(
                parent, child,
                get_edge_visits_.get_edge_visits(parent, child) + 1);
            ++it;
            ++next_it;
        }
    }

    current_node_ = root_;
    backprop_path_.clear();
    sim_length_  = 0;
    in_rollout_  = false;
}

template<typename INodeHandle, typename IChoice, typename IFloat,
         typename IGetVisits, typename IGetValue,
         typename ISetVisits, typename ISetValue,
         typename IWalker, typename IGetArrivalReward,
         typename IGetEdgeVisits, typename ISetEdgeVisits,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IRolloutChoose>
size_t
sim<INodeHandle, IChoice, IFloat,
    IGetVisits, IGetValue, ISetVisits, ISetValue,
    IWalker, IGetArrivalReward,
    IGetEdgeVisits, ISetEdgeVisits,
    IGetChoiceCount, IGetChoiceAt,
    IRolloutChoose>::length() const
{
    return sim_length_;
}

} // namespace monte_carlo

#endif // SIM_HPP
