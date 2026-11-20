#ifndef BEST_PATH_COMMIT_VALUE_MANIFEST_HPP
#define BEST_PATH_COMMIT_VALUE_MANIFEST_HPP

#include "infrastructure/best_path_child_finder.hpp"
#include "infrastructure/best_path_recorder.hpp"
#include "infrastructure/best_path_recording_terminator.hpp"
#include "infrastructure/choice_logger.hpp"
#include "infrastructure/committed_policy.hpp"
#include "infrastructure/committing_policy.hpp"
#include "infrastructure/episode_choice_log.hpp"
#include "infrastructure/in_rollout_flag.hpp"
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
#include "infrastructure/visit_threshold.hpp"

namespace monte_carlo
{

// best_path_commit_value_manifest
//
// UCT with commitments along the best episode found so far: every choice of an
// episode, in the tree and in its rollout, is logged (choice_logger); an
// episode whose reward beats the best so far has its path recorded
// (best_path_recorder); and once the frontier has gathered a set number of
// visits, the search commits to the choice the best episode made there
// (best_path_child_finder). Selection and backups are UCT's. The commitments
// and the best path are kept outside, like the visit and value tables.

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
    typename IFindCommitment,
    typename IGetFrontier,
    typename IGetFrontierBaseVisits,
    typename ICommit,
    typename IFindStep,
    typename IGetBestReward,
    typename IStartPath,
    typename IAddStep,
    typename IWalker,
    typename IGetChoiceCount,
    typename IGetChoiceAt,
    typename IRndGen
>
struct best_path_commit_value_manifest
{
    using rollout_t        = random_rollout<IChoice, IRndGen, IGetChoiceCount, IGetChoiceAt>;
    using exploration_t    = uniform_exploration_constant<IFloat>;
    using delta_t          = uniform_value_delta<IFloat>;
    using value_update_t   = uniform_value_update<INodeHandle, IGetValue, ISetValue, delta_t>;
    using cursor_t         = uct_cursor<INodeHandle>;
    using path_t           = uct_backprop_path<INodeHandle>;
    using visit_creditor_t = uct_visit_creditor<INodeHandle, path_t, IGetVisits, ISetVisits>;
    using value_creditor_t = uct_value_creditor<visit_creditor_t, path_t, value_update_t>;
    using threshold_t      = visit_threshold;
    using ucb1_t           = ucb1<INodeHandle, IChoice, IFloat,
                                  IGetVisits, IGetValue, IIncludeValue, INormalizeValue, IWalker,
                                  exploration_t, IGetChoiceCount, IGetChoiceAt>;
    using child_finder_t   = best_path_child_finder<INodeHandle, IChoice, IFindStep,
                                                    IGetChoiceCount, IGetChoiceAt>;
    using committing_t     = committing_policy<INodeHandle, IChoice, IGetVisits, IWalker,
                                               IGetFrontier, IGetFrontierBaseVisits, threshold_t, child_finder_t,
                                               ICommit, ucb1_t, IGetChoiceCount, IGetChoiceAt>;
    using policy_t         = committed_policy<INodeHandle, IChoice, IFindCommitment, committing_t,
                                              IGetChoiceCount, IGetChoiceAt>;
    using uct_chooser_t    = uct_chooser<INodeHandle, IChoice, IGetVisits, IWalker,
                                         IGetChoiceCount, IGetChoiceAt,
                                         policy_t, rollout_t,
                                         cursor_t, cursor_t, path_t,
                                         in_rollout_flag, in_rollout_flag>;
    using choice_log_t     = episode_choice_log<IChoice>;
    using chooser_t        = choice_logger<IChoice, IGetChoiceCount, IGetChoiceAt, uct_chooser_t, choice_log_t>;
    using uct_terminator_t = uct_terminator<INodeHandle,
                                            path_t,
                                            value_creditor_t,
                                            path_t, path_t,
                                            cursor_t,
                                            in_rollout_flag>;
    using recorder_t       = best_path_recorder<INodeHandle, IChoice, IFloat, delta_t,
                                                choice_log_t, choice_log_t,
                                                IGetBestReward, IStartPath, IAddStep, IWalker>;
    using terminator_t     = best_path_recording_terminator<uct_terminator_t, recorder_t>;

    best_path_commit_value_manifest(IGetVisits&             get_visits,
                                    ISetVisits&             set_visits,
                                    IGetValue&              get_value,
                                    ISetValue&              set_value,
                                    IIncludeValue&          include_value,
                                    INormalizeValue&        normalize_value,
                                    IFindCommitment&        find_commitment,
                                    IGetFrontier&           get_frontier,
                                    IGetFrontierBaseVisits& get_frontier_base_visits,
                                    ICommit&                commit,
                                    IFindStep&              find_step,
                                    IGetBestReward&         get_best_reward,
                                    IStartPath&             start_path,
                                    IAddStep&               add_step,
                                    IRndGen&                rnd_gen,
                                    IFloat                  exploration_constant,
                                    size_t                  commit_visits,
                                    INodeHandle             root);

    IWalker          walker;
    rollout_t        rollout;
    exploration_t    exploration_constant;
    delta_t          delta;
    value_update_t   value_update;
    cursor_t         cursor;
    path_t           backprop_path;
    visit_creditor_t visit_creditor;
    value_creditor_t value_creditor;
    threshold_t      threshold;
    child_finder_t   child_finder;
    ucb1_t           ucb1_policy;
    committing_t     committing;
    policy_t         policy;
    in_rollout_flag  in_rollout;
    uct_chooser_t    tree_chooser;
    choice_log_t     choice_log;
    chooser_t        chooser;
    uct_terminator_t backprop_terminator;
    recorder_t       recorder;
    terminator_t     terminator;
};

template<typename INH, typename IC, typename IF,
         typename IGVis, typename ISVis, typename IGVal, typename ISVal, typename IIV, typename INV,
         typename IFC, typename IGF, typename IGFBV, typename ICm,
         typename IFS, typename IGBR, typename ISP, typename IAS,
         typename IW, typename IGCC, typename IGCA, typename IRG>
best_path_commit_value_manifest<INH, IC, IF, IGVis, ISVis, IGVal, ISVal, IIV, INV, IFC, IGF, IGFBV, ICm,
                                IFS, IGBR, ISP, IAS, IW, IGCC, IGCA, IRG>::best_path_commit_value_manifest(
        IGVis& get_visits,
        ISVis& set_visits,
        IGVal& get_value,
        ISVal& set_value,
        IIV&   include_value,
        INV&   normalize_value,
        IFC&   find_commitment,
        IGF&   get_frontier,
        IGFBV& get_frontier_base_visits,
        ICm&   commit,
        IFS&   find_step,
        IGBR&  get_best_reward,
        ISP&   start_path,
        IAS&   add_step,
        IRG&   rnd_gen,
        IF     exploration_constant,
        size_t commit_visits,
        INH    root)
    : walker()
    , rollout(rnd_gen)
    , exploration_constant(exploration_constant)
    , delta()
    , value_update(get_value, set_value, delta)
    , cursor(root)
    , backprop_path(root)
    , visit_creditor(backprop_path, get_visits, set_visits)
    , value_creditor(visit_creditor, backprop_path, value_update)
    , threshold(commit_visits)
    , child_finder(find_step)
    , ucb1_policy(get_visits, get_value, include_value, normalize_value, walker, this->exploration_constant)
    , committing(get_visits, walker, get_frontier, get_frontier_base_visits, threshold, child_finder, commit,
                 ucb1_policy)
    , policy(find_commitment, committing)
    , in_rollout()
    , tree_chooser(get_visits, walker, policy, rollout,
                   cursor, cursor, backprop_path,
                   in_rollout, in_rollout)
    , choice_log()
    , chooser(tree_chooser, choice_log)
    , backprop_terminator(backprop_path, value_creditor, backprop_path, backprop_path,
                          cursor, in_rollout, root)
    , recorder(delta, choice_log, choice_log, get_best_reward, start_path, add_step, walker, root)
    , terminator(backprop_terminator, recorder)
{}

}

#endif
