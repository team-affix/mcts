#ifndef DBUCT_VALUE_CREDITOR_HPP
#define DBUCT_VALUE_CREDITOR_HPP

namespace monte_carlo
{

template<
    typename ICreditVisit,
    typename IGetTopValueFrame,
    typename IAddValue,
    typename IGetValueDelta
>
struct dbuct_value_creditor
{
    dbuct_value_creditor(ICreditVisit&      visit_creditor,
                         IGetTopValueFrame& get_top_value_frame,
                         IAddValue&         value_adder,
                         IGetValueDelta&    value_delta);

    void credit();

private:
    ICreditVisit&      visit_creditor_;
    IGetTopValueFrame& get_top_value_frame_;
    IAddValue&         value_adder_;
    IGetValueDelta&    value_delta_;
};

template<typename IVC, typename IGTVF, typename IVA, typename IGVD>
dbuct_value_creditor<IVC, IGTVF, IVA, IGVD>::dbuct_value_creditor(
        IVC&   visit_creditor,
        IGTVF& get_top_value_frame,
        IVA&   value_adder,
        IGVD&  value_delta)
    : visit_creditor_(visit_creditor)
    , get_top_value_frame_(get_top_value_frame)
    , value_adder_(value_adder)
    , value_delta_(value_delta)
{}

template<typename IVC, typename IGTVF, typename IVA, typename IGVD>
void dbuct_value_creditor<IVC, IGTVF, IVA, IGVD>::credit()
{
    visit_creditor_.credit();
    value_adder_.add_value(value_delta_.get_value_delta(get_top_value_frame_.top().handle));
}

}

#endif
