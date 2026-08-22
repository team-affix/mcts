#ifndef UCT_VALUE_CREDITOR_HPP
#define UCT_VALUE_CREDITOR_HPP

namespace monte_carlo
{

template<
    typename INodeHandle,
    typename ICreditVisit,
    typename IUpdateNode
>
struct uct_value_creditor
{
    uct_value_creditor(ICreditVisit& visit_creditor,
                       IUpdateNode&  update_node);

    void credit(const INodeHandle& node);

private:
    ICreditVisit& visit_creditor_;
    IUpdateNode&  update_node_;
};

template<typename INH, typename ICV, typename IUN>
uct_value_creditor<INH, ICV, IUN>::uct_value_creditor(
        ICV& visit_creditor,
        IUN& update_node)
    : visit_creditor_(visit_creditor)
    , update_node_(update_node)
{}

template<typename INH, typename ICV, typename IUN>
void uct_value_creditor<INH, ICV, IUN>::credit(const INH& node)
{
    visit_creditor_.credit(node);
    update_node_.update(node);
}

}

#endif
