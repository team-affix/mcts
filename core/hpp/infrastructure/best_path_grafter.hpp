#ifndef BEST_PATH_GRAFTER_HPP
#define BEST_PATH_GRAFTER_HPP

#include <limits>

namespace monte_carlo
{

// best_path_grafter
//
// Ends the rollout and, when the episode's reward is at least the best reward
// grafted so far, grafts the rollout's path onto the tree: it replays the
// logged rollout choices from the last node on the backprop path, pushing each
// node reached, so crediting the path adds every one of them to the tree. A
// worse episode grows the tree by one node, as in UCT. Either way the log is
// cleared. This is CadiaPlayer's single-agent enhancement (Bjornsson and
// Finnsson, 2009): a good solution, once found, is never forgotten.

template<
    typename INodeHandle,
    typename IChoice,
    typename IFloat,
    typename IExitRollout,
    typename IGetReward,
    typename IGetChoices,
    typename IClearChoices,
    typename IGetTopNode,
    typename IWalker,
    typename IPushNode
>
struct best_path_grafter
{
    best_path_grafter(IExitRollout&  exit_rollout,
                      IGetReward&    get_reward,
                      IGetChoices&   get_choices,
                      IClearChoices& clear_choices,
                      IGetTopNode&   get_top_node,
                      IWalker&       walker,
                      IPushNode&     push_node);

    void graft();

private:
    IExitRollout&  exit_rollout_;
    IGetReward&    get_reward_;
    IGetChoices&   get_choices_;
    IClearChoices& clear_choices_;
    IGetTopNode&   get_top_node_;
    IWalker&       walker_;
    IPushNode&     push_node_;
    IFloat         best_reward_;
};

template<typename INH, typename IC, typename IF, typename IER, typename IGR,
         typename IGC, typename ICC, typename IGTN, typename IW, typename IPN>
best_path_grafter<INH, IC, IF, IER, IGR, IGC, ICC, IGTN, IW, IPN>::best_path_grafter(
        IER&  exit_rollout,
        IGR&  get_reward,
        IGC&  get_choices,
        ICC&  clear_choices,
        IGTN& get_top_node,
        IW&   walker,
        IPN&  push_node)
    : exit_rollout_(exit_rollout)
    , get_reward_(get_reward)
    , get_choices_(get_choices)
    , clear_choices_(clear_choices)
    , get_top_node_(get_top_node)
    , walker_(walker)
    , push_node_(push_node)
    , best_reward_(-std::numeric_limits<IF>::infinity())
{}

template<typename INH, typename IC, typename IF, typename IER, typename IGR,
         typename IGC, typename ICC, typename IGTN, typename IW, typename IPN>
void best_path_grafter<INH, IC, IF, IER, IGR, IGC, ICC, IGTN, IW, IPN>::graft()
{
    exit_rollout_.exit_rollout();
    const IF   reward = get_reward_.value();
    const bool worse  = reward < best_reward_;
    if (worse)
    {
        clear_choices_.clear_choices();
        return;
    }
    best_reward_ = reward;
    INH node     = get_top_node_.top();
    for (const IC& choice : get_choices_.choices())
    {
        node = walker_.walk(node, choice);
        push_node_.push(node);
    }
    clear_choices_.clear_choices();
}

}

#endif
