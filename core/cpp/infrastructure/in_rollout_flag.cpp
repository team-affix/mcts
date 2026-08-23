#include "infrastructure/in_rollout_flag.hpp"

namespace monte_carlo
{

in_rollout_flag::in_rollout_flag()
    : in_rollout_(false)
{}

bool in_rollout_flag::is_in_rollout() const
{
    return in_rollout_;
}

void in_rollout_flag::enter_rollout()
{
    in_rollout_ = true;
}

void in_rollout_flag::exit_rollout()
{
    in_rollout_ = false;
}

}
