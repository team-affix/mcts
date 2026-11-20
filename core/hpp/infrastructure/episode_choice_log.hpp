#ifndef EPISODE_CHOICE_LOG_HPP
#define EPISODE_CHOICE_LOG_HPP

#include <vector>

namespace monte_carlo
{

// episode_choice_log
//
// Every choice made in the current episode, in the tree and in its rollout, in
// the order they were made.

template<typename IChoice>
struct episode_choice_log
{
    episode_choice_log();

    void                        push_choice(const IChoice& choice);
    const std::vector<IChoice>& choices() const;
    void                        clear_choices();

private:
    std::vector<IChoice> choices_;
};

template<typename IChoice>
episode_choice_log<IChoice>::episode_choice_log()
    : choices_()
{}

template<typename IChoice>
void episode_choice_log<IChoice>::push_choice(const IChoice& choice)
{
    choices_.push_back(choice);
}

template<typename IChoice>
const std::vector<IChoice>& episode_choice_log<IChoice>::choices() const
{
    return choices_;
}

template<typename IChoice>
void episode_choice_log<IChoice>::clear_choices()
{
    choices_.clear();
}

}

#endif
