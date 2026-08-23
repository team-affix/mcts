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

}

#endif
