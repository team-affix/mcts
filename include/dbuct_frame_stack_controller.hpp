#ifndef DBUCT_FRAME_STACK_CONTROLLER_HPP
#define DBUCT_FRAME_STACK_CONTROLLER_HPP

#include <cstddef>
#include "dbuct_frame.hpp"

namespace monte_carlo
{

template<
    typename INodeHandle,
    typename IPushFrame,
    typename IPopFrame,
    typename IGetTopFrame,
    typename IAddVisits
>
struct dbuct_frame_stack_controller
{
    dbuct_frame_stack_controller(IPushFrame&   push_frame,
                                 IPopFrame&    pop_frame,
                                 IGetTopFrame& get_top_frame,
                                 IAddVisits&   add_visits);

    void forestep(const dbuct_frame<INodeHandle>& f);
    void backstep();

private:
    IPushFrame&   push_frame_;
    IPopFrame&    pop_frame_;
    IGetTopFrame& get_top_frame_;
    IAddVisits&   add_visits_;
};

template<typename INH, typename IPuF, typename IPoF, typename IGTF, typename IAV>
dbuct_frame_stack_controller<INH, IPuF, IPoF, IGTF, IAV>::dbuct_frame_stack_controller(
        IPuF& push_frame,
        IPoF& pop_frame,
        IGTF& get_top_frame,
        IAV&  add_visits)
    : push_frame_(push_frame)
    , pop_frame_(pop_frame)
    , get_top_frame_(get_top_frame)
    , add_visits_(add_visits)
{}

template<typename INH, typename IPuF, typename IPoF, typename IGTF, typename IAV>
void dbuct_frame_stack_controller<INH, IPuF, IPoF, IGTF, IAV>::forestep(
        const dbuct_frame<INH>& f)
{
    push_frame_.push(f);
}

template<typename INH, typename IPuF, typename IPoF, typename IGTF, typename IAV>
void dbuct_frame_stack_controller<INH, IPuF, IPoF, IGTF, IAV>::backstep()
{
    size_t v = get_top_frame_.top().visit_lump;
    pop_frame_.pop();
    add_visits_.add_visits(v);
}

}

#endif
