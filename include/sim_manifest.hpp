#ifndef SIM_MANIFEST_HPP
#define SIM_MANIFEST_HPP

#include "in_rollout_flag.hpp"
#include "random_rollout.hpp"
#include "sim.hpp"
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
struct sim_manifest
{
    using rollout_t      = random_rollout<IChoice, IRndGen, IGetChoiceCount, IGetChoiceAt>;
    using exploration_t  = uniform_exploration_constant<IFloat>;
    using delta_t        = uniform_value_delta<IFloat>;
    using value_update_t = uniform_value_update<INodeHandle, IGetValue, ISetValue, delta_t>;
    using policy_t       = ucb1<INodeHandle, IChoice, IFloat,
                                IGetVisits, IGetValue, IWalker,
                                exploration_t, IGetChoiceCount, IGetChoiceAt>;
    using sim_t          = sim<INodeHandle, IChoice,
                               IGetVisits, ISetVisits,
                               IWalker,
                               IGetChoiceCount, IGetChoiceAt,
                               policy_t, rollout_t, value_update_t,
                               in_rollout_flag, in_rollout_flag>;

    sim_manifest(IGetVisits& get_visits,
                 ISetVisits& set_visits,
                 IGetValue&  get_value,
                 ISetValue&  set_value,
                 IRndGen&    rnd_gen,
                 IFloat      exploration_constant,
                 INodeHandle root);

    IWalker        walker;
    rollout_t      rollout;
    exploration_t  exploration_constant;
    delta_t        delta;
    value_update_t value_update;
    policy_t       policy;
    in_rollout_flag in_rollout;
    sim_t          s;
};

template<typename INH, typename IC, typename IF,
         typename IGVis, typename ISVis, typename IGVal, typename ISVal,
         typename IW, typename IGCC, typename IGCA, typename IRG>
sim_manifest<INH, IC, IF, IGVis, ISVis, IGVal, ISVal, IW, IGCC, IGCA, IRG>::sim_manifest(
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
    , policy(get_visits, get_value, walker, this->exploration_constant)
    , in_rollout()
    , s(get_visits, set_visits, walker, policy, rollout, value_update,
        in_rollout, in_rollout, root)
{}

}

#endif
