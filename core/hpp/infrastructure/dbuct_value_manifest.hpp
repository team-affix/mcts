#ifndef DBUCT_VALUE_MANIFEST_HPP
#define DBUCT_VALUE_MANIFEST_HPP

#include <cstddef>
#include <limits>
#include "infrastructure/dbuct_chooser.hpp"
#include "infrastructure/dbuct_frame_stack.hpp"
#include "infrastructure/dbuct_frame_stack_controller.hpp"
#include "infrastructure/dbuct_terminator.hpp"
#include "infrastructure/dbuct_value_adder.hpp"
#include "infrastructure/dbuct_value_creditor.hpp"
#include "infrastructure/dbuct_value_stack.hpp"
#include "infrastructure/dbuct_value_stack_controller.hpp"
#include "infrastructure/dbuct_visit_adder.hpp"
#include "infrastructure/dbuct_visit_creditor.hpp"
#include "infrastructure/dispatches_table.hpp"
#include "infrastructure/in_rollout_flag.hpp"
#include "infrastructure/linear_batch_increment.hpp"
#include "infrastructure/random_rollout.hpp"
#include "infrastructure/ucb1.hpp"
#include "infrastructure/uniform_exploration_constant.hpp"
#include "infrastructure/uniform_value_delta.hpp"
#include "value_objects/dbuct_frame.hpp"
#include "value_objects/dbuct_value_frame.hpp"

namespace monte_carlo
{

template<
    typename INodeHandle,
    typename IChoice,
    typename IFloat,
    typename IGetVisits,
    typename ISetVisits,
    typename IGetValue,
    typename ISetValue,
    typename IWalker,
    typename IGetChoiceCount,
    typename IGetChoiceAt,
    typename IRndGen,
    template<typename...> typename Map
>
struct dbuct_value_manifest
{
    using rollout_t       = random_rollout<IChoice, IRndGen, IGetChoiceCount, IGetChoiceAt>;
    using exploration_t   = uniform_exploration_constant<IFloat>;
    using delta_t         = uniform_value_delta<IFloat>;
    using dispatches_t    = dispatches_table<INodeHandle, Map>;
    using frame_stack_t   = dbuct_frame_stack<INodeHandle>;
    using visit_adder_t   = dbuct_visit_adder<INodeHandle, frame_stack_t,
                                              IGetVisits, ISetVisits>;
    using frame_stack_controller_t
                          = dbuct_frame_stack_controller<INodeHandle,
                                                         frame_stack_t, frame_stack_t,
                                                         frame_stack_t, visit_adder_t>;
    using value_stack_t   = dbuct_value_stack<INodeHandle, IFloat>;
    using value_adder_t   = dbuct_value_adder<INodeHandle, IFloat, value_stack_t,
                                              IGetValue, ISetValue>;
    using value_stack_controller_t
                          = dbuct_value_stack_controller<INodeHandle, IFloat,
                                                         frame_stack_controller_t,
                                                         frame_stack_controller_t,
                                                         value_stack_t, value_stack_t,
                                                         value_stack_t, value_adder_t>;
    using visit_creditor_t = dbuct_visit_creditor<visit_adder_t>;
    using value_creditor_t = dbuct_value_creditor<visit_creditor_t, value_stack_t,
                                                  value_adder_t, delta_t>;
    using policy_t        = ucb1<INodeHandle, IChoice, IFloat,
                                 IGetVisits, IGetValue, IWalker,
                                 exploration_t, IGetChoiceCount, IGetChoiceAt>;
    using chooser_t       = dbuct_chooser<INodeHandle, IChoice,
                                          IGetVisits,
                                          dispatches_t, dispatches_t, linear_batch_increment,
                                          value_stack_controller_t,
                                          frame_stack_t,
                                          IWalker,
                                          IGetChoiceCount, IGetChoiceAt,
                                          policy_t, rollout_t,
                                          in_rollout_flag, in_rollout_flag>;
    using terminator_t    = dbuct_terminator<value_stack_controller_t,
                                             frame_stack_t,
                                             value_creditor_t,
                                             in_rollout_flag>;

    dbuct_value_manifest(IGetVisits& get_visits,
                         ISetVisits& set_visits,
                         IGetValue&  get_value,
                         ISetValue&  set_value,
                         IRndGen&    rnd_gen,
                         IFloat      exploration_constant,
                         size_t      grant_increment_interval,
                         INodeHandle root);

    IWalker                   walker;
    rollout_t                 rollout;
    linear_batch_increment    batch;
    exploration_t             exploration_constant;
    delta_t                   delta;
    dispatches_t              dispatches;
    frame_stack_t             frame_stack;
    visit_adder_t             visit_adder;
    frame_stack_controller_t  frame_stack_controller;
    value_stack_t             value_stack;
    value_adder_t             value_adder;
    value_stack_controller_t  value_stack_controller;
    visit_creditor_t          visit_creditor;
    value_creditor_t          value_creditor;
    policy_t                  policy;
    in_rollout_flag           in_rollout;
    chooser_t                 chooser;
    terminator_t              terminator;
};

template<typename INH, typename IC, typename IF,
         typename IGVis, typename ISVis, typename IGVal, typename ISVal,
         typename IW, typename IGCC, typename IGCA, typename IRG,
         template<typename...> typename Map>
dbuct_value_manifest<INH, IC, IF, IGVis, ISVis, IGVal, ISVal, IW, IGCC, IGCA, IRG, Map>::dbuct_value_manifest(
        IGVis& get_visits,
        ISVis& set_visits,
        IGVal& get_value,
        ISVal& set_value,
        IRG&   rnd_gen,
        IF     exploration_constant,
        size_t grant_increment_interval,
        INH    root)
    : walker()
    , rollout(rnd_gen)
    , batch(grant_increment_interval)
    , exploration_constant(exploration_constant)
    , delta()
    , dispatches()
    , frame_stack(dbuct_frame<INH>(root, std::numeric_limits<size_t>::max()))
    , visit_adder(frame_stack, get_visits, set_visits)
    , frame_stack_controller(frame_stack, frame_stack, frame_stack, visit_adder)
    , value_stack(dbuct_value_frame<INH, IF>(root))
    , value_adder(value_stack, get_value, set_value)
    , value_stack_controller(frame_stack_controller, frame_stack_controller,
                             value_stack, value_stack, value_stack, value_adder)
    , visit_creditor(visit_adder)
    , value_creditor(visit_creditor, value_stack, value_adder, delta)
    , policy(get_visits, get_value, walker, this->exploration_constant)
    , in_rollout()
    , chooser(get_visits, dispatches, dispatches, batch,
              value_stack_controller, frame_stack,
              walker, policy, rollout,
              in_rollout, in_rollout)
    , terminator(value_stack_controller, frame_stack, value_creditor, in_rollout)
{}

}

#endif
