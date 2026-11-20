#ifndef FOCUS_VALUE_MANIFEST_HPP
#define FOCUS_VALUE_MANIFEST_HPP

#include "infrastructure/focus_ucb1.hpp"
#include "infrastructure/in_rollout_flag.hpp"
#include "infrastructure/power_band.hpp"
#include "infrastructure/random_rollout.hpp"
#include "infrastructure/uct_backprop_path.hpp"
#include "infrastructure/uct_chooser.hpp"
#include "infrastructure/uct_cursor.hpp"
#include "infrastructure/uct_terminator.hpp"
#include "infrastructure/uct_value_creditor.hpp"
#include "infrastructure/uct_visit_creditor.hpp"
#include "infrastructure/uniform_exploration_constant.hpp"
#include "infrastructure/uniform_focus_bonus.hpp"
#include "infrastructure/uniform_value_delta.hpp"
#include "infrastructure/uniform_value_update.hpp"

namespace monte_carlo
{

// focus_value_manifest
//
// uct_value_manifest with focus_ucb1 as the selection policy: among the
// children tied with the best within a band A * N^(-r) of the node's visits N,
// the first in index order gets a constant bonus. Everything else is UCT's.
// Like the visit and value tables, the normalization bounds are kept outside.

template<
    typename INodeHandle,
    typename IChoice,
    typename IFloat,
    typename IGetVisits,
    typename ISetVisits,
    typename IGetValue,
    typename ISetValue,
    typename IIncludeValue,
    typename INormalizeValue,
    typename IWalker,
    typename IGetChoiceCount,
    typename IGetChoiceAt,
    typename IRndGen
>
struct focus_value_manifest
{
    using rollout_t        = random_rollout<IChoice, IRndGen, IGetChoiceCount, IGetChoiceAt>;
    using exploration_t    = uniform_exploration_constant<IFloat>;
    using delta_t          = uniform_value_delta<IFloat>;
    using value_update_t   = uniform_value_update<INodeHandle, IGetValue, ISetValue, delta_t>;
    using cursor_t         = uct_cursor<INodeHandle>;
    using path_t           = uct_backprop_path<INodeHandle>;
    using visit_creditor_t = uct_visit_creditor<INodeHandle, path_t, IGetVisits, ISetVisits>;
    using value_creditor_t = uct_value_creditor<visit_creditor_t, path_t, value_update_t>;
    using band_t           = power_band<IFloat>;
    using bonus_t          = uniform_focus_bonus<IFloat>;
    using policy_t         = focus_ucb1<INodeHandle, IChoice, IFloat,
                                        IGetVisits, IGetValue, IIncludeValue, INormalizeValue, IWalker,
                                        exploration_t, band_t, bonus_t, IGetChoiceCount, IGetChoiceAt>;
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

    focus_value_manifest(IGetVisits&      get_visits,
                         ISetVisits&      set_visits,
                         IGetValue&       get_value,
                         ISetValue&       set_value,
                         IIncludeValue&   include_value,
                         INormalizeValue& normalize_value,
                         IRndGen&         rnd_gen,
                         IFloat           exploration_constant,
                         IFloat           focus_width,
                         IFloat           focus_rate,
                         IFloat           focus_bonus,
                         INodeHandle      root);

    IWalker          walker;
    rollout_t        rollout;
    exploration_t    exploration_constant;
    band_t           band;
    bonus_t          bonus;
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
         typename IGVis, typename ISVis, typename IGVal, typename ISVal, typename IIV, typename INV,
         typename IW, typename IGCC, typename IGCA, typename IRG>
focus_value_manifest<INH, IC, IF, IGVis, ISVis, IGVal, ISVal, IIV, INV, IW, IGCC, IGCA, IRG>::focus_value_manifest(
        IGVis& get_visits,
        ISVis& set_visits,
        IGVal& get_value,
        ISVal& set_value,
        IIV&   include_value,
        INV&   normalize_value,
        IRG&   rnd_gen,
        IF     exploration_constant,
        IF     focus_width,
        IF     focus_rate,
        IF     focus_bonus,
        INH    root)
    : walker()
    , rollout(rnd_gen)
    , exploration_constant(exploration_constant)
    , band(focus_width, focus_rate)
    , bonus(focus_bonus)
    , delta()
    , value_update(get_value, set_value, delta)
    , cursor(root)
    , backprop_path(root)
    , visit_creditor(backprop_path, get_visits, set_visits)
    , value_creditor(visit_creditor, backprop_path, value_update)
    , policy(get_visits, get_value, include_value, normalize_value, walker, this->exploration_constant, band, bonus)
    , in_rollout()
    , chooser(get_visits, walker, policy, rollout,
              cursor, cursor, backprop_path,
              in_rollout, in_rollout)
    , terminator(backprop_path, value_creditor, backprop_path, backprop_path,
                 cursor, in_rollout, root)
{}

}

#endif
