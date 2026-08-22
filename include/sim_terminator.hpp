#ifndef SIM_TERMINATOR_HPP
#define SIM_TERMINATOR_HPP

namespace monte_carlo
{

template<
    typename INodeHandle,
    typename IGetNodeCount,
    typename IGetTopNode,
    typename IPopNode,
    typename ICreditNode,
    typename ISetInRollout
>
struct sim_terminator
{
    sim_terminator(IGetNodeCount& get_node_count,
                   IGetTopNode&   get_top_node,
                   IPopNode&      pop_node,
                   ICreditNode&   credit_node,
                   ISetInRollout& set_in_rollout);

    void terminate();

private:
    IGetNodeCount& get_node_count_;
    IGetTopNode&   get_top_node_;
    IPopNode&      pop_node_;
    ICreditNode&   credit_node_;
    ISetInRollout& set_in_rollout_;
};

template<typename INH, typename IGNC, typename IGTN, typename IPN,
         typename ICN, typename ISIR>
sim_terminator<INH, IGNC, IGTN, IPN, ICN, ISIR>::sim_terminator(
        IGNC& get_node_count,
        IGTN& get_top_node,
        IPN&  pop_node,
        ICN&  credit_node,
        ISIR& set_in_rollout)
    : get_node_count_(get_node_count)
    , get_top_node_(get_top_node)
    , pop_node_(pop_node)
    , credit_node_(credit_node)
    , set_in_rollout_(set_in_rollout)
{}

template<typename INH, typename IGNC, typename IGTN, typename IPN,
         typename ICN, typename ISIR>
void sim_terminator<INH, IGNC, IGTN, IPN, ICN, ISIR>::terminate()
{
    while (get_node_count_.size() > 0)
    {
        credit_node_.credit(get_top_node_.top());
        pop_node_.pop();
    }

    set_in_rollout_.set_in_rollout(false);
}

}

#endif
