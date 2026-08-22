#ifndef UCT_TERMINATOR_HPP
#define UCT_TERMINATOR_HPP

namespace monte_carlo
{

template<
    typename INodeHandle,
    typename IGetNodeCount,
    typename ICredit,
    typename IPopNode,
    typename IPushNode,
    typename ISetCurrentNode,
    typename IExitRollout
>
struct uct_terminator
{
    uct_terminator(IGetNodeCount&   get_node_count,
                   ICredit&         creditor,
                   IPopNode&        pop_node,
                   IPushNode&       push_node,
                   ISetCurrentNode& set_current_node,
                   IExitRollout&    exit_rollout,
                   INodeHandle      root);

    void terminate();

private:
    IGetNodeCount&   get_node_count_;
    ICredit&         creditor_;
    IPopNode&        pop_node_;
    IPushNode&       push_node_;
    ISetCurrentNode& set_current_node_;
    IExitRollout&    exit_rollout_;
    INodeHandle      root_;
};

template<typename INH, typename IGNC, typename ICr, typename IPoN, typename IPuN,
         typename ISCN, typename IER>
uct_terminator<INH, IGNC, ICr, IPoN, IPuN, ISCN, IER>::uct_terminator(
        IGNC& get_node_count,
        ICr&  creditor,
        IPoN& pop_node,
        IPuN& push_node,
        ISCN& set_current_node,
        IER&  exit_rollout,
        INH   root)
    : get_node_count_(get_node_count)
    , creditor_(creditor)
    , pop_node_(pop_node)
    , push_node_(push_node)
    , set_current_node_(set_current_node)
    , exit_rollout_(exit_rollout)
    , root_(root)
{}

template<typename INH, typename IGNC, typename ICr, typename IPoN, typename IPuN,
         typename ISCN, typename IER>
void uct_terminator<INH, IGNC, ICr, IPoN, IPuN, ISCN, IER>::terminate()
{
    while (get_node_count_.size() > 0)
    {
        creditor_.credit();
        pop_node_.pop();
    }

    push_node_.push(root_);
    set_current_node_.set_current_node(root_);

    exit_rollout_.exit_rollout();
}

}

#endif
