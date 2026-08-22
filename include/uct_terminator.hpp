#ifndef UCT_TERMINATOR_HPP
#define UCT_TERMINATOR_HPP

namespace monte_carlo
{

template<
    typename INodeHandle,
    typename IGetNodeCount,
    typename IGetTopNode,
    typename IPopNode,
    typename IPushNode,
    typename ICreditNode,
    typename ISetCurrentNode,
    typename IExitRollout
>
struct uct_terminator
{
    uct_terminator(IGetNodeCount&   get_node_count,
                   IGetTopNode&     get_top_node,
                   IPopNode&        pop_node,
                   IPushNode&       push_node,
                   ICreditNode&     credit_node,
                   ISetCurrentNode& set_current_node,
                   IExitRollout&    exit_rollout,
                   INodeHandle      root);

    void terminate();

private:
    IGetNodeCount&   get_node_count_;
    IGetTopNode&     get_top_node_;
    IPopNode&        pop_node_;
    IPushNode&       push_node_;
    ICreditNode&     credit_node_;
    ISetCurrentNode& set_current_node_;
    IExitRollout&    exit_rollout_;
    INodeHandle      root_;
};

template<typename INH, typename IGNC, typename IGTN, typename IPoN, typename IPuN,
         typename ICN, typename ISCN, typename IER>
uct_terminator<INH, IGNC, IGTN, IPoN, IPuN, ICN, ISCN, IER>::uct_terminator(
        IGNC& get_node_count,
        IGTN& get_top_node,
        IPoN& pop_node,
        IPuN& push_node,
        ICN&  credit_node,
        ISCN& set_current_node,
        IER&  exit_rollout,
        INH   root)
    : get_node_count_(get_node_count)
    , get_top_node_(get_top_node)
    , pop_node_(pop_node)
    , push_node_(push_node)
    , credit_node_(credit_node)
    , set_current_node_(set_current_node)
    , exit_rollout_(exit_rollout)
    , root_(root)
{}

template<typename INH, typename IGNC, typename IGTN, typename IPoN, typename IPuN,
         typename ICN, typename ISCN, typename IER>
void uct_terminator<INH, IGNC, IGTN, IPoN, IPuN, ICN, ISCN, IER>::terminate()
{
    while (get_node_count_.size() > 0)
    {
        credit_node_.credit(get_top_node_.top());
        pop_node_.pop();
    }

    push_node_.push(root_);
    set_current_node_.set_current_node(root_);

    exit_rollout_.exit_rollout();
}

}

#endif
