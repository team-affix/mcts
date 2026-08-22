#ifndef UCT_VISIT_CREDITOR_HPP
#define UCT_VISIT_CREDITOR_HPP

namespace monte_carlo
{

template<
    typename INodeHandle,
    typename IGetTopNode,
    typename IGetVisits,
    typename ISetVisits
>
struct uct_visit_creditor
{
    uct_visit_creditor(IGetTopNode& get_top_node,
                       IGetVisits&  get_visits,
                       ISetVisits&  set_visits);

    void credit();

private:
    IGetTopNode& get_top_node_;
    IGetVisits&  get_visits_;
    ISetVisits&  set_visits_;
};

template<typename INH, typename IGTN, typename IGVis, typename ISVis>
uct_visit_creditor<INH, IGTN, IGVis, ISVis>::uct_visit_creditor(
        IGTN&  get_top_node,
        IGVis& get_visits,
        ISVis& set_visits)
    : get_top_node_(get_top_node)
    , get_visits_(get_visits)
    , set_visits_(set_visits)
{}

template<typename INH, typename IGTN, typename IGVis, typename ISVis>
void uct_visit_creditor<INH, IGTN, IGVis, ISVis>::credit()
{
    INH node = get_top_node_.top();
    set_visits_.set_visits(node, get_visits_.get_visits(node) + 1);
}

}

#endif
