#ifndef SIM_HPP
#define SIM_HPP

#include <cstddef>
#include <vector>

namespace monte_carlo
{

template<
    typename INodeHandle,
    typename IChoice,
    typename IGetVisits,
    typename ISetVisits,
    typename IWalker,
    typename IGetChoiceCount,
    typename IGetChoiceAt,
    typename IPolicyChoose,
    typename IRolloutChoose,
    typename IUpdateNode
>
struct sim
{
    sim(IGetVisits&     get_visits,
        ISetVisits&     set_visits,
        IWalker&        walker,
        IPolicyChoose&  policy,
        IRolloutChoose& rollout,
        IUpdateNode&    update_node,
        INodeHandle     root);

    IChoice choose(const IGetChoiceCount& get_choice_count, const IGetChoiceAt& get_choice_at);
    void    terminate();
    size_t  length() const;

private:
    IGetVisits&     get_visits_;
    ISetVisits&     set_visits_;
    IWalker&        walker_;
    IPolicyChoose&  policy_;
    IRolloutChoose& rollout_;
    IUpdateNode&    update_node_;

    INodeHandle              current_node_;
    std::vector<INodeHandle> backprop_path_;
    size_t                   sim_length_;
    bool                     in_rollout_;
};

template<typename INodeHandle, typename IChoice,
         typename IGetVisits, typename ISetVisits,
         typename IWalker,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IPolicyChoose, typename IRolloutChoose,
         typename IUpdateNode>
sim<INodeHandle, IChoice,
    IGetVisits, ISetVisits,
    IWalker,
    IGetChoiceCount, IGetChoiceAt,
    IPolicyChoose, IRolloutChoose,
    IUpdateNode>::sim(
        IGetVisits&     get_visits,
        ISetVisits&     set_visits,
        IWalker&        walker,
        IPolicyChoose&  policy,
        IRolloutChoose& rollout,
        IUpdateNode&    update_node,
        INodeHandle     root)
    : get_visits_(get_visits)
    , set_visits_(set_visits)
    , walker_(walker)
    , policy_(policy)
    , rollout_(rollout)
    , update_node_(update_node)
    , current_node_(root)
    , backprop_path_({root})
    , sim_length_(0)
    , in_rollout_(false)
{}

template<typename INodeHandle, typename IChoice,
         typename IGetVisits, typename ISetVisits,
         typename IWalker,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IPolicyChoose, typename IRolloutChoose,
         typename IUpdateNode>
IChoice
sim<INodeHandle, IChoice,
    IGetVisits, ISetVisits,
    IWalker,
    IGetChoiceCount, IGetChoiceAt,
    IPolicyChoose, IRolloutChoose,
    IUpdateNode>::choose(
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

    IChoice     chosen       = policy_.policy_choose(current_node_, get_choice_count, get_choice_at);
    INodeHandle chosen_child = walker_.walk(current_node_, chosen);
    backprop_path_.push_back(chosen_child);
    current_node_ = chosen_child;

    size_t child_visits = get_visits_.get_visits(chosen_child);

    if (child_visits == 0)
        in_rollout_ = true;

    return chosen;
}

template<typename INodeHandle, typename IChoice,
         typename IGetVisits, typename ISetVisits,
         typename IWalker,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IPolicyChoose, typename IRolloutChoose,
         typename IUpdateNode>
void
sim<INodeHandle, IChoice,
    IGetVisits, ISetVisits,
    IWalker,
    IGetChoiceCount, IGetChoiceAt,
    IPolicyChoose, IRolloutChoose,
    IUpdateNode>::terminate()
{
    for (const INodeHandle& node : backprop_path_)
    {
        set_visits_.set_visits(node, get_visits_.get_visits(node) + 1);
        update_node_.update(node);
    }
}

template<typename INodeHandle, typename IChoice,
         typename IGetVisits, typename ISetVisits,
         typename IWalker,
         typename IGetChoiceCount, typename IGetChoiceAt,
         typename IPolicyChoose, typename IRolloutChoose,
         typename IUpdateNode>
size_t
sim<INodeHandle, IChoice,
    IGetVisits, ISetVisits,
    IWalker,
    IGetChoiceCount, IGetChoiceAt,
    IPolicyChoose, IRolloutChoose,
    IUpdateNode>::length() const
{
    return sim_length_;
}

}

#endif
