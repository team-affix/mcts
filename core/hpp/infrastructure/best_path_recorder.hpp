#ifndef BEST_PATH_RECORDER_HPP
#define BEST_PATH_RECORDER_HPP

namespace monte_carlo
{

// best_path_recorder
//
// Ends an episode's path record: when the episode's reward beats the best so
// far, it replaces the best path with this episode's, walking the logged
// choices from the root and storing the choice made at each node. Either way it
// clears the log. It walks with the rollout over, so every node is named.

template<
    typename INodeHandle,
    typename IChoice,
    typename IFloat,
    typename IGetReward,
    typename IGetChoices,
    typename IClearChoices,
    typename IGetBestReward,
    typename IStartPath,
    typename IAddStep,
    typename IWalker
>
struct best_path_recorder
{
    best_path_recorder(IGetReward&     get_reward,
                       IGetChoices&    get_choices,
                       IClearChoices&  clear_choices,
                       IGetBestReward& get_best_reward,
                       IStartPath&     start_path,
                       IAddStep&       add_step,
                       IWalker&        walker,
                       INodeHandle     root);

    void record();

private:
    IGetReward&     get_reward_;
    IGetChoices&    get_choices_;
    IClearChoices&  clear_choices_;
    IGetBestReward& get_best_reward_;
    IStartPath&     start_path_;
    IAddStep&       add_step_;
    IWalker&        walker_;
    INodeHandle     root_;
};

template<typename INH, typename IC, typename IF, typename IGR, typename IGC, typename ICC,
         typename IGBR, typename ISP, typename IAS, typename IW>
best_path_recorder<INH, IC, IF, IGR, IGC, ICC, IGBR, ISP, IAS, IW>::best_path_recorder(
        IGR&  get_reward,
        IGC&  get_choices,
        ICC&  clear_choices,
        IGBR& get_best_reward,
        ISP&  start_path,
        IAS&  add_step,
        IW&   walker,
        INH   root)
    : get_reward_(get_reward)
    , get_choices_(get_choices)
    , clear_choices_(clear_choices)
    , get_best_reward_(get_best_reward)
    , start_path_(start_path)
    , add_step_(add_step)
    , walker_(walker)
    , root_(root)
{}

template<typename INH, typename IC, typename IF, typename IGR, typename IGC, typename ICC,
         typename IGBR, typename ISP, typename IAS, typename IW>
void best_path_recorder<INH, IC, IF, IGR, IGC, ICC, IGBR, ISP, IAS, IW>::record()
{
    const IF   reward = get_reward_.value();
    const bool better = reward > get_best_reward_.best_reward();
    if (!better)
    {
        clear_choices_.clear_choices();
        return;
    }
    start_path_.start_path(reward);
    INH node = root_;
    for (const IC& choice : get_choices_.choices())
    {
        add_step_.add_step(node, choice);
        node = walker_.walk(node, choice);
    }
    clear_choices_.clear_choices();
}

}

#endif
