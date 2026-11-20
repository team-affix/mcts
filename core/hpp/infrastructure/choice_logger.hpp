#ifndef CHOICE_LOGGER_HPP
#define CHOICE_LOGGER_HPP

namespace monte_carlo
{

// choice_logger
//
// A chooser that chooses through the chooser it wraps and logs every choice it
// returns, so an episode's whole path, tree and rollout alike, can be replayed.

template<
    typename IChoice,
    typename IGetChoiceCount,
    typename IGetChoiceAt,
    typename IChoose,
    typename IPushChoice
>
struct choice_logger
{
    choice_logger(IChoose&     chooser,
                  IPushChoice& push_choice);

    IChoice choose(const IGetChoiceCount& get_choice_count,
                   const IGetChoiceAt&    get_choice_at);

private:
    IChoose&     chooser_;
    IPushChoice& push_choice_;
};

template<typename IC, typename IGCC, typename IGCA, typename ICh, typename IPC>
choice_logger<IC, IGCC, IGCA, ICh, IPC>::choice_logger(
        ICh& chooser,
        IPC& push_choice)
    : chooser_(chooser)
    , push_choice_(push_choice)
{}

template<typename IC, typename IGCC, typename IGCA, typename ICh, typename IPC>
IC choice_logger<IC, IGCC, IGCA, ICh, IPC>::choose(
        const IGCC& get_choice_count,
        const IGCA& get_choice_at)
{
    const IC chosen = chooser_.choose(get_choice_count, get_choice_at);
    push_choice_.push_choice(chosen);
    return chosen;
}

}

#endif
