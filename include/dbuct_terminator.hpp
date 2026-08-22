#ifndef DBUCT_TERMINATOR_HPP
#define DBUCT_TERMINATOR_HPP

#include <cstddef>

namespace monte_carlo
{

template<
    typename IBackstep,
    typename IGetTopFrame,
    typename ICredit,
    typename IExitRollout
>
struct dbuct_terminator
{
    dbuct_terminator(IBackstep&    backstep,
                     IGetTopFrame& get_top_frame,
                     ICredit&      creditor,
                     IExitRollout& exit_rollout);

    void terminate();

private:
    IBackstep&    backstep_;
    IGetTopFrame& get_top_frame_;
    ICredit&      creditor_;
    IExitRollout& exit_rollout_;
};

template<typename IBa, typename IGTF, typename ICr, typename IER>
dbuct_terminator<IBa, IGTF, ICr, IER>::dbuct_terminator(
        IBa&  backstep,
        IGTF& get_top_frame,
        ICr&  creditor,
        IER&  exit_rollout)
    : backstep_(backstep)
    , get_top_frame_(get_top_frame)
    , creditor_(creditor)
    , exit_rollout_(exit_rollout)
{}

template<typename IBa, typename IGTF, typename ICr, typename IER>
void dbuct_terminator<IBa, IGTF, ICr, IER>::terminate()
{
    creditor_.credit();

    while (get_top_frame_.top().visit_lump >= get_top_frame_.top().budget)
        backstep_.backstep();

    exit_rollout_.exit_rollout();
}

}

#endif
