#ifndef DBUCT_ADD_VISITS_HPP
#define DBUCT_ADD_VISITS_HPP

#include <cstddef>
#include "dbuct_frame.hpp"

namespace monte_carlo
{

template<
    typename INodeHandle,
    typename IGetTopFrame,
    typename IGetVisits,
    typename ISetVisits
>
struct dbuct_add_visits
{
    dbuct_add_visits(IGetTopFrame& get_top_frame,
                     IGetVisits&   get_visits,
                     ISetVisits&   set_visits);

    void add_visits(size_t v);

private:
    IGetTopFrame& get_top_frame_;
    IGetVisits&   get_visits_;
    ISetVisits&   set_visits_;
};

template<typename INH, typename IGTF, typename IGVis, typename ISVis>
dbuct_add_visits<INH, IGTF, IGVis, ISVis>::dbuct_add_visits(
        IGTF&  get_top_frame,
        IGVis& get_visits,
        ISVis& set_visits)
    : get_top_frame_(get_top_frame)
    , get_visits_(get_visits)
    , set_visits_(set_visits)
{}

template<typename INH, typename IGTF, typename IGVis, typename ISVis>
void dbuct_add_visits<INH, IGTF, IGVis, ISVis>::add_visits(size_t v)
{
    dbuct_frame<INH>& f = get_top_frame_.top();
    set_visits_.set_visits(f.handle, get_visits_.get_visits(f.handle) + v);
    f.visit_lump += v;
}

}

#endif
