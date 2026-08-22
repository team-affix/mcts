#ifndef IN_ROLLOUT_FLAG_HPP
#define IN_ROLLOUT_FLAG_HPP

namespace monte_carlo
{

struct in_rollout_flag
{
    in_rollout_flag();

    bool is_in_rollout() const;
    void enter_rollout();
    void exit_rollout();

private:
    bool in_rollout_;
};

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

#endif
