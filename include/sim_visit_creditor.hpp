#ifndef SIM_VISIT_CREDITOR_HPP
#define SIM_VISIT_CREDITOR_HPP

namespace monte_carlo
{

template<
    typename INodeHandle,
    typename IGetVisits,
    typename ISetVisits
>
struct sim_visit_creditor
{
    sim_visit_creditor(IGetVisits& get_visits,
                       ISetVisits& set_visits);

    void credit(const INodeHandle& node);

private:
    IGetVisits& get_visits_;
    ISetVisits& set_visits_;
};

template<typename INH, typename IGVis, typename ISVis>
sim_visit_creditor<INH, IGVis, ISVis>::sim_visit_creditor(
        IGVis& get_visits,
        ISVis& set_visits)
    : get_visits_(get_visits)
    , set_visits_(set_visits)
{}

template<typename INH, typename IGVis, typename ISVis>
void sim_visit_creditor<INH, IGVis, ISVis>::credit(const INH& node)
{
    set_visits_.set_visits(node, get_visits_.get_visits(node) + 1);
}

}

#endif
