#ifndef ROLLOUT_CHOICE_LOG_HPP
#define ROLLOUT_CHOICE_LOG_HPP

#include <vector>

namespace monte_carlo
{

// rollout_choice_log
//
// The choices made in the current episode's rollout, in the order they were
// made.

template<typename IChoice>
struct rollout_choice_log
{
    rollout_choice_log();

    void                        push_choice(const IChoice& choice);
    const std::vector<IChoice>& choices() const;
    void                        clear_choices();

private:
    std::vector<IChoice> choices_;
};

template<typename IChoice>
rollout_choice_log<IChoice>::rollout_choice_log()
    : choices_()
{}

template<typename IChoice>
void rollout_choice_log<IChoice>::push_choice(const IChoice& choice)
{
    choices_.push_back(choice);
}

template<typename IChoice>
const std::vector<IChoice>& rollout_choice_log<IChoice>::choices() const
{
    return choices_;
}

template<typename IChoice>
void rollout_choice_log<IChoice>::clear_choices()
{
    choices_.clear();
}

}

#endif
