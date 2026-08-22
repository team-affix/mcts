#ifndef UCT_VISIT_CREDITOR_HPP
#define UCT_VISIT_CREDITOR_HPP

namespace monte_carlo
{

template<
    typename INodeHandle,
    typename IGetVisits,
    typename ISetVisits
>
struct uct_visit_creditor
{
    uct_visit_creditor(IGetVisits& get_visits,
                       ISetVisits& set_visits);

    void credit(const INodeHandle& node);

private:
    IGetVisits& get_visits_;
    ISetVisits& set_visits_;
};

template<typename INH, typename IGVis, typename ISVis>
uct_visit_creditor<INH, IGVis, ISVis>::uct_visit_creditor(
        IGVis& get_visits,
        ISVis& set_visits)
    : get_visits_(get_visits)
    , set_visits_(set_visits)
{}

template<typename INH, typename IGVis, typename ISVis>
void uct_visit_creditor<INH, IGVis, ISVis>::credit(const INH& node)
{
    set_visits_.set_visits(node, get_visits_.get_visits(node) + 1);
}

}

#endif
