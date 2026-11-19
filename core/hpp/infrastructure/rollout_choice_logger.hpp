#ifndef ROLLOUT_CHOICE_LOGGER_HPP
#define ROLLOUT_CHOICE_LOGGER_HPP

namespace monte_carlo
{

// rollout_choice_logger
//
// A rollout policy that chooses through another rollout policy and logs each
// choice it makes, so the rollout's path can be replayed into the tree later.

template<
    typename IChoice,
    typename IGetChoiceCount,
    typename IGetChoiceAt,
    typename IRolloutChoose,
    typename IPushChoice
>
struct rollout_choice_logger
{
    rollout_choice_logger(IRolloutChoose& rollout,
                          IPushChoice&    push_choice);

    IChoice rollout_choose(const IGetChoiceCount& get_choice_count,
                           const IGetChoiceAt&    get_choice_at);

private:
    IRolloutChoose& rollout_;
    IPushChoice&    push_choice_;
};

template<typename IC, typename IGCC, typename IGCA, typename IRC, typename IPC>
rollout_choice_logger<IC, IGCC, IGCA, IRC, IPC>::rollout_choice_logger(
        IRC& rollout,
        IPC& push_choice)
    : rollout_(rollout)
    , push_choice_(push_choice)
{}

template<typename IC, typename IGCC, typename IGCA, typename IRC, typename IPC>
IC rollout_choice_logger<IC, IGCC, IGCA, IRC, IPC>::rollout_choose(
        const IGCC& get_choice_count,
        const IGCA& get_choice_at)
{
    const IC chosen = rollout_.rollout_choose(get_choice_count, get_choice_at);
    push_choice_.push_choice(chosen);
    return chosen;
}

}

#endif
