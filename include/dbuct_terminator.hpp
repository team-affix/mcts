#ifndef DBUCT_TERMINATOR_HPP
#define DBUCT_TERMINATOR_HPP

#include <cstddef>

namespace monte_carlo
{

template<
    typename IBackstep,
    typename IGetTopFrame,
    typename IValueCreditor,
    typename IExitRollout
>
struct dbuct_terminator
{
    dbuct_terminator(IBackstep&      backstep,
                     IGetTopFrame&   get_top_frame,
                     IValueCreditor& value_creditor,
                     IExitRollout&   exit_rollout);

    void terminate();

private:
    IBackstep&      backstep_;
    IGetTopFrame&   get_top_frame_;
    IValueCreditor& value_creditor_;
    IExitRollout&   exit_rollout_;
};

template<typename IBa, typename IGTF, typename IVC, typename IER>
dbuct_terminator<IBa, IGTF, IVC, IER>::dbuct_terminator(
        IBa&  backstep,
        IGTF& get_top_frame,
        IVC&  value_creditor,
        IER&  exit_rollout)
    : backstep_(backstep)
    , get_top_frame_(get_top_frame)
    , value_creditor_(value_creditor)
    , exit_rollout_(exit_rollout)
{}

template<typename IBa, typename IGTF, typename IVC, typename IER>
void dbuct_terminator<IBa, IGTF, IVC, IER>::terminate()
{
    value_creditor_.credit();

    while (get_top_frame_.top().visit_lump >= get_top_frame_.top().budget)
        backstep_.backstep();

    exit_rollout_.exit_rollout();
}

}

#endif
