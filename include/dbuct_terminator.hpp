#ifndef DBUCT_TERMINATOR_HPP
#define DBUCT_TERMINATOR_HPP

#include <cstddef>

namespace monte_carlo
{

template<
    typename IBackstep,
    typename IGetTopFrame,
    typename IValueCreditor,
    typename ISetInRollout
>
struct dbuct_terminator
{
    dbuct_terminator(IBackstep&        backstep,
                     IGetTopFrame&     get_top_frame,
                     IValueCreditor&   value_creditor,
                     ISetInRollout&    set_in_rollout);

    void terminate();

private:
    IBackstep&      backstep_;
    IGetTopFrame&   get_top_frame_;
    IValueCreditor& value_creditor_;
    ISetInRollout&  set_in_rollout_;
};

template<typename IBa, typename IGTF, typename IVC, typename ISIR>
dbuct_terminator<IBa, IGTF, IVC, ISIR>::dbuct_terminator(
        IBa&  backstep,
        IGTF& get_top_frame,
        IVC&  value_creditor,
        ISIR& set_in_rollout)
    : backstep_(backstep)
    , get_top_frame_(get_top_frame)
    , value_creditor_(value_creditor)
    , set_in_rollout_(set_in_rollout)
{}

template<typename IBa, typename IGTF, typename IVC, typename ISIR>
void dbuct_terminator<IBa, IGTF, IVC, ISIR>::terminate()
{
    value_creditor_.credit();

    while (get_top_frame_.top().visit_lump >= get_top_frame_.top().budget)
        backstep_.backstep();

    set_in_rollout_.set_in_rollout(false);
}

}

#endif
