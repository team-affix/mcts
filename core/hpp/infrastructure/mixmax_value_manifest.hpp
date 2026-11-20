#ifndef MIXMAX_VALUE_MANIFEST_HPP
#define MIXMAX_VALUE_MANIFEST_HPP

#include "infrastructure/in_rollout_flag.hpp"
#include "infrastructure/max_value_update.hpp"
#include "infrastructure/mixmax_value_view.hpp"
#include "infrastructure/paired_value_update.hpp"
#include "infrastructure/random_rollout.hpp"
#include "infrastructure/ucb1.hpp"
#include "infrastructure/uct_backprop_path.hpp"
#include "infrastructure/uct_chooser.hpp"
#include "infrastructure/uct_cursor.hpp"
#include "infrastructure/uct_terminator.hpp"
#include "infrastructure/uct_value_creditor.hpp"
#include "infrastructure/uct_visit_creditor.hpp"
#include "infrastructure/uniform_exploration_constant.hpp"
#include "infrastructure/uniform_value_delta.hpp"
#include "infrastructure/uniform_value_update.hpp"

namespace monte_carlo
{

// mixmax_value_manifest
//
// uct_value_manifest whose selection reads each child as a blend of its mean
// and the best reward backed up through it: (1 - w) * mean + w * best. Each
// backup adds the reward to the node's sum and raises its best. A choice whose
// subtree holds one good path among many bad ones so keeps a value near that
// path's instead of the average. Like the visit and value tables, the best
// rewards and the normalization bounds are kept outside.

template<
    typename INodeHandle,
    typename IChoice,
    typename IFloat,
    typename IGetVisits,
    typename ISetVisits,
    typename IGetValue,
    typename ISetValue,
    typename IGetBest,
    typename ISetBest,
    typename IIncludeValue,
    typename INormalizeValue,
    typename IWalker,
    typename IGetChoiceCount,
    typename IGetChoiceAt,
    typename IRndGen
>
struct mixmax_value_manifest
{
    using rollout_t        = random_rollout<IChoice, IRndGen, IGetChoiceCount, IGetChoiceAt>;
    using exploration_t    = uniform_exploration_constant<IFloat>;
    using delta_t          = uniform_value_delta<IFloat>;
    using sum_update_t     = uniform_value_update<INodeHandle, IGetValue, ISetValue, delta_t>;
    using best_update_t    = max_value_update<INodeHandle, IGetVisits, IGetBest, ISetBest, delta_t>;
    using value_update_t   = paired_value_update<INodeHandle, sum_update_t, best_update_t>;
    using view_t           = mixmax_value_view<INodeHandle, IFloat, IGetVisits, IGetValue, IGetBest>;
    using cursor_t         = uct_cursor<INodeHandle>;
    using path_t           = uct_backprop_path<INodeHandle>;
    using visit_creditor_t = uct_visit_creditor<INodeHandle, path_t, IGetVisits, ISetVisits>;
    using value_creditor_t = uct_value_creditor<visit_creditor_t, path_t, value_update_t>;
    using policy_t         = ucb1<INodeHandle, IChoice, IFloat,
                                  IGetVisits, view_t, IIncludeValue, INormalizeValue, IWalker,
                                  exploration_t, IGetChoiceCount, IGetChoiceAt>;
    using chooser_t        = uct_chooser<INodeHandle, IChoice, IGetVisits, IWalker,
                                         IGetChoiceCount, IGetChoiceAt,
                                         policy_t, rollout_t,
                                         cursor_t, cursor_t, path_t,
                                         in_rollout_flag, in_rollout_flag>;
    using terminator_t     = uct_terminator<INodeHandle,
                                            path_t,
                                            value_creditor_t,
                                            path_t, path_t,
                                            cursor_t,
                                            in_rollout_flag>;

    mixmax_value_manifest(IGetVisits&      get_visits,
                          ISetVisits&      set_visits,
                          IGetValue&       get_value,
                          ISetValue&       set_value,
                          IGetBest&        get_best,
                          ISetBest&        set_best,
                          IIncludeValue&   include_value,
                          INormalizeValue& normalize_value,
                          IRndGen&         rnd_gen,
                          IFloat           exploration_constant,
                          IFloat           best_weight,
                          INodeHandle      root);

    IWalker          walker;
    rollout_t        rollout;
    exploration_t    exploration_constant;
    delta_t          delta;
    sum_update_t     sum_update;
    best_update_t    best_update;
    value_update_t   value_update;
    view_t           view;
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
         typename IGVis, typename ISVis, typename IGVal, typename ISVal, typename IGB, typename ISB,
         typename IIV, typename INV, typename IW, typename IGCC, typename IGCA, typename IRG>
mixmax_value_manifest<INH, IC, IF, IGVis, ISVis, IGVal, ISVal, IGB, ISB, IIV, INV, IW, IGCC, IGCA, IRG>::
    mixmax_value_manifest(
        IGVis& get_visits,
        ISVis& set_visits,
        IGVal& get_value,
        ISVal& set_value,
        IGB&   get_best,
        ISB&   set_best,
        IIV&   include_value,
        INV&   normalize_value,
        IRG&   rnd_gen,
        IF     exploration_constant,
        IF     best_weight,
        INH    root)
    : walker()
    , rollout(rnd_gen)
    , exploration_constant(exploration_constant)
    , delta()
    , sum_update(get_value, set_value, delta)
    , best_update(get_visits, get_best, set_best, delta)
    , value_update(sum_update, best_update)
    , view(get_visits, get_value, get_best, best_weight)
    , cursor(root)
    , backprop_path(root)
    , visit_creditor(backprop_path, get_visits, set_visits)
    , value_creditor(visit_creditor, backprop_path, value_update)
    , policy(get_visits, view, include_value, normalize_value, walker, this->exploration_constant)
    , in_rollout()
    , chooser(get_visits, walker, policy, rollout,
              cursor, cursor, backprop_path,
              in_rollout, in_rollout)
    , terminator(backprop_path, value_creditor, backprop_path, backprop_path,
                 cursor, in_rollout, root)
{}

}

#endif
