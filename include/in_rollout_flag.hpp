#ifndef IN_ROLLOUT_FLAG_HPP
#define IN_ROLLOUT_FLAG_HPP

namespace monte_carlo
{

struct in_rollout_flag
{
    in_rollout_flag();

    bool get_in_rollout() const;
    void set_in_rollout(bool v);

private:
    bool in_rollout_;
};

in_rollout_flag::in_rollout_flag()
    : in_rollout_(false)
{}

bool in_rollout_flag::get_in_rollout() const
{
    return in_rollout_;
}

void in_rollout_flag::set_in_rollout(bool v)
{
    in_rollout_ = v;
}

}

#endif
