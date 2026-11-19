#ifndef BEST_PATH_VALUE_MANIFEST_HPP
#define BEST_PATH_VALUE_MANIFEST_HPP

#include "infrastructure/best_path_grafter.hpp"
#include "infrastructure/best_path_terminator.hpp"
#include "infrastructure/in_rollout_flag.hpp"
#include "infrastructure/random_rollout.hpp"
#include "infrastructure/rollout_choice_log.hpp"
#include "infrastructure/rollout_choice_logger.hpp"
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

// best_path_value_manifest
//
// uct_value_manifest with best-path memory for single-player search: the
// rollout's choices are logged, and an episode whose reward is at least the
// best so far has its whole path, rollout included, added to the tree
// (best_path_grafter). Selection and backups are UCT's: nodes average their
// rewards and children are scored by ucb1.

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
struct best_path_value_manifest
{
    using rollout_t             = random_rollout<IChoice, IRndGen, IGetChoiceCount, IGetChoiceAt>;
    using choice_log_t          = rollout_choice_log<IChoice>;
    using rollout_logger_t      = rollout_choice_logger<IChoice, IGetChoiceCount, IGetChoiceAt,
                                                        rollout_t, choice_log_t>;
    using exploration_t         = uniform_exploration_constant<IFloat>;
    using delta_t               = uniform_value_delta<IFloat>;
    using value_update_t        = uniform_value_update<INodeHandle, IGetValue, ISetValue, delta_t>;
    using cursor_t              = uct_cursor<INodeHandle>;
    using path_t                = uct_backprop_path<INodeHandle>;
    using visit_creditor_t      = uct_visit_creditor<INodeHandle, path_t, IGetVisits, ISetVisits>;
    using value_creditor_t      = uct_value_creditor<visit_creditor_t, path_t, value_update_t>;
    using policy_t              = ucb1<INodeHandle, IChoice, IFloat,
                                       IGetVisits, IGetValue, IIncludeValue, INormalizeValue, IWalker,
                                       exploration_t, IGetChoiceCount, IGetChoiceAt>;
    using chooser_t             = uct_chooser<INodeHandle, IChoice, IGetVisits, IWalker,
                                              IGetChoiceCount, IGetChoiceAt,
                                              policy_t, rollout_logger_t,
                                              cursor_t, cursor_t, path_t,
                                              in_rollout_flag, in_rollout_flag>;
    using grafter_t             = best_path_grafter<INodeHandle, IChoice, IFloat,
                                                    in_rollout_flag, delta_t,
                                                    choice_log_t, choice_log_t,
                                                    path_t, IWalker, path_t>;
    using backprop_terminator_t = uct_terminator<INodeHandle,
                                                 path_t,
                                                 value_creditor_t,
                                                 path_t, path_t,
                                                 cursor_t,
                                                 in_rollout_flag>;
    using terminator_t          = best_path_terminator<grafter_t, backprop_terminator_t>;

    best_path_value_manifest(IGetVisits&      get_visits,
                             ISetVisits&      set_visits,
                             IGetValue&       get_value,
                             ISetValue&       set_value,
                             IIncludeValue&   include_value,
                             INormalizeValue& normalize_value,
                             IRndGen&         rnd_gen,
                             IFloat           exploration_constant,
                             INodeHandle      root);

    IWalker               walker;
    rollout_t             rollout;
    choice_log_t          choice_log;
    rollout_logger_t      rollout_logger;
    exploration_t         exploration_constant;
    delta_t               delta;
    value_update_t        value_update;
    cursor_t              cursor;
    path_t                backprop_path;
    visit_creditor_t      visit_creditor;
    value_creditor_t      value_creditor;
    policy_t              policy;
    in_rollout_flag       in_rollout;
    chooser_t             chooser;
    grafter_t             grafter;
    backprop_terminator_t backprop_terminator;
    terminator_t          terminator;
};

template<typename INH, typename IC, typename IF,
         typename IGVis, typename ISVis, typename IGVal, typename ISVal, typename IIV, typename INV,
         typename IW, typename IGCC, typename IGCA, typename IRG>
best_path_value_manifest<INH, IC, IF, IGVis, ISVis, IGVal, ISVal, IIV, INV, IW, IGCC, IGCA, IRG>::best_path_value_manifest(
        IGVis& get_visits,
        ISVis& set_visits,
        IGVal& get_value,
        ISVal& set_value,
        IIV&   include_value,
        INV&   normalize_value,
        IRG&   rnd_gen,
        IF     exploration_constant,
        INH    root)
    : walker()
    , rollout(rnd_gen)
    , choice_log()
    , rollout_logger(rollout, choice_log)
    , exploration_constant(exploration_constant)
    , delta()
    , value_update(get_value, set_value, delta)
    , cursor(root)
    , backprop_path(root)
    , visit_creditor(backprop_path, get_visits, set_visits)
    , value_creditor(visit_creditor, backprop_path, value_update)
    , policy(get_visits, get_value, include_value, normalize_value, walker, this->exploration_constant)
    , in_rollout()
    , chooser(get_visits, walker, policy, rollout_logger,
              cursor, cursor, backprop_path,
              in_rollout, in_rollout)
    , grafter(in_rollout, delta, choice_log, choice_log, backprop_path, walker, backprop_path)
    , backprop_terminator(backprop_path, value_creditor, backprop_path, backprop_path,
                          cursor, in_rollout, root)
    , terminator(grafter, backprop_terminator)
{}

}

#endif
