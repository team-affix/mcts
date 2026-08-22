#ifndef SIM_VALUE_MANIFEST_HPP
#define SIM_VALUE_MANIFEST_HPP

#include "in_rollout_flag.hpp"
#include "random_rollout.hpp"
#include "sim_backprop_path.hpp"
#include "sim_chooser.hpp"
#include "sim_cursor.hpp"
#include "sim_terminator.hpp"
#include "sim_value_creditor.hpp"
#include "sim_visit_creditor.hpp"
#include "ucb1.hpp"
#include "uniform_exploration_constant.hpp"
#include "uniform_value_delta.hpp"
#include "uniform_value_update.hpp"

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
    typename IRndGen
>
struct sim_value_manifest
{
    using rollout_t        = random_rollout<IChoice, IRndGen, IGetChoiceCount, IGetChoiceAt>;
    using exploration_t    = uniform_exploration_constant<IFloat>;
    using delta_t          = uniform_value_delta<IFloat>;
    using value_update_t   = uniform_value_update<INodeHandle, IGetValue, ISetValue, delta_t>;
    using cursor_t         = sim_cursor<INodeHandle>;
    using path_t           = sim_backprop_path<INodeHandle>;
    using visit_creditor_t = sim_visit_creditor<INodeHandle, IGetVisits, ISetVisits>;
    using value_creditor_t = sim_value_creditor<INodeHandle, visit_creditor_t, value_update_t>;
    using policy_t         = ucb1<INodeHandle, IChoice, IFloat,
                                  IGetVisits, IGetValue, IWalker,
                                  exploration_t, IGetChoiceCount, IGetChoiceAt>;
    using chooser_t        = sim_chooser<INodeHandle, IChoice, IGetVisits, IWalker,
                                         IGetChoiceCount, IGetChoiceAt,
                                         policy_t, rollout_t,
                                         cursor_t, cursor_t, path_t,
                                         in_rollout_flag, in_rollout_flag>;
    using terminator_t     = sim_terminator<INodeHandle, path_t, path_t, path_t,
                                            value_creditor_t, in_rollout_flag>;

    sim_value_manifest(IGetVisits& get_visits,
                       ISetVisits& set_visits,
                       IGetValue&  get_value,
                       ISetValue&  set_value,
                       IRndGen&    rnd_gen,
                       IFloat      exploration_constant,
                       INodeHandle root);

    IWalker          walker;
    rollout_t        rollout;
    exploration_t    exploration_constant;
    delta_t          delta;
    value_update_t   value_update;
    cursor_t         cursor;
    path_t           backprop_path;
    visit_creditor_t visit_creditor;
    value_creditor_t value_creditor;
    policy_t         policy;
    in_rollout_flag  in_rollout;
    chooser_t        chooser;
    terminator_t     terminator;
};

template<typename INH, typename IC, typename IF,
         typename IGVis, typename ISVis, typename IGVal, typename ISVal,
         typename IW, typename IGCC, typename IGCA, typename IRG>
sim_value_manifest<INH, IC, IF, IGVis, ISVis, IGVal, ISVal, IW, IGCC, IGCA, IRG>::sim_value_manifest(
        IGVis& get_visits,
        ISVis& set_visits,
        IGVal& get_value,
        ISVal& set_value,
        IRG&   rnd_gen,
        IF     exploration_constant,
        INH    root)
    : walker()
    , rollout(rnd_gen)
    , exploration_constant(exploration_constant)
    , delta()
    , value_update(get_value, set_value, delta)
    , cursor(root)
    , backprop_path(root)
    , visit_creditor(get_visits, set_visits)
    , value_creditor(visit_creditor, value_update)
    , policy(get_visits, get_value, walker, this->exploration_constant)
    , in_rollout()
    , chooser(get_visits, walker, policy, rollout,
              cursor, cursor, backprop_path,
              in_rollout, in_rollout)
    , terminator(backprop_path, backprop_path, backprop_path,
                 value_creditor, in_rollout)
{}

}

#endif
