#ifndef DBUCT_VALUE_STACK_CONTROLLER_HPP
#define DBUCT_VALUE_STACK_CONTROLLER_HPP

#include "value_objects/dbuct_frame.hpp"
#include "value_objects/dbuct_value_frame.hpp"

namespace monte_carlo
{

template<
    typename INodeHandle,
    typename IFloat,
    typename IForestep,
    typename IBackstep,
    typename IPushValueFrame,
    typename IPopValueFrame,
    typename IGetTopValueFrame,
    typename IAddValue
>
struct dbuct_value_stack_controller
{
    dbuct_value_stack_controller(IForestep&         forestep,
                                 IBackstep&         backstep,
                                 IPushValueFrame&   push_value_frame,
                                 IPopValueFrame&    pop_value_frame,
                                 IGetTopValueFrame& get_top_value_frame,
                                 IAddValue&         add_value);

    void forestep(const dbuct_frame<INodeHandle>& f);
    void backstep();

private:
    IForestep&         forestep_;
    IBackstep&         backstep_;
    IPushValueFrame&   push_value_frame_;
    IPopValueFrame&    pop_value_frame_;
    IGetTopValueFrame& get_top_value_frame_;
    IAddValue&         add_value_;
};

template<typename INH, typename IF, typename IFo, typename IBa,
         typename IPuVF, typename IPoVF, typename IGTVF, typename IAV>
dbuct_value_stack_controller<INH, IF, IFo, IBa, IPuVF, IPoVF, IGTVF, IAV>::dbuct_value_stack_controller(
        IFo&   forestep,
        IBa&   backstep,
        IPuVF& push_value_frame,
        IPoVF& pop_value_frame,
        IGTVF& get_top_value_frame,
        IAV&   add_value)
    : forestep_(forestep)
    , backstep_(backstep)
    , push_value_frame_(push_value_frame)
    , pop_value_frame_(pop_value_frame)
    , get_top_value_frame_(get_top_value_frame)
    , add_value_(add_value)
{}

template<typename INH, typename IF, typename IFo, typename IBa,
         typename IPuVF, typename IPoVF, typename IGTVF, typename IAV>
void dbuct_value_stack_controller<INH, IF, IFo, IBa, IPuVF, IPoVF, IGTVF, IAV>::forestep(
        const dbuct_frame<INH>& f)
{
    forestep_.forestep(f);
    push_value_frame_.push(dbuct_value_frame<INH, IF>(f.handle));
}

template<typename INH, typename IF, typename IFo, typename IBa,
         typename IPuVF, typename IPoVF, typename IGTVF, typename IAV>
void dbuct_value_stack_controller<INH, IF, IFo, IBa, IPuVF, IPoVF, IGTVF, IAV>::backstep()
{
    IF l = get_top_value_frame_.top().value_lump;
    pop_value_frame_.pop();
    backstep_.backstep();
    add_value_.add_value(l);
}

}

#endif
