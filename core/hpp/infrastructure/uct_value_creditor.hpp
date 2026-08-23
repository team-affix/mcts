#ifndef UCT_VALUE_CREDITOR_HPP
#define UCT_VALUE_CREDITOR_HPP

namespace monte_carlo
{

template<
    typename ICreditVisit,
    typename IGetTopNode,
    typename IUpdateNode
>
struct uct_value_creditor
{
    uct_value_creditor(ICreditVisit& visit_creditor,
                       IGetTopNode&  get_top_node,
                       IUpdateNode&  update_node);

    void credit();

private:
    ICreditVisit& visit_creditor_;
    IGetTopNode&  get_top_node_;
    IUpdateNode&  update_node_;
};

template<typename ICV, typename IGTN, typename IUN>
uct_value_creditor<ICV, IGTN, IUN>::uct_value_creditor(
        ICV&  visit_creditor,
        IGTN& get_top_node,
        IUN&  update_node)
    : visit_creditor_(visit_creditor)
    , get_top_node_(get_top_node)
    , update_node_(update_node)
{}

template<typename ICV, typename IGTN, typename IUN>
void uct_value_creditor<ICV, IGTN, IUN>::credit()
{
    visit_creditor_.credit();
    update_node_.update(get_top_node_.top());
}

}

#endif
