#ifndef DBUCT_VALUE_TERMINATOR_HPP
#define DBUCT_VALUE_TERMINATOR_HPP

namespace monte_carlo
{

template<
    typename ITerminate,
    typename IGetTopValueFrame,
    typename IAddValue,
    typename IGetValueDelta
>
struct dbuct_value_terminator
{
    dbuct_value_terminator(ITerminate&        terminate,
                           IGetTopValueFrame& get_top_value_frame,
                           IAddValue&         add_value,
                           IGetValueDelta&    value_delta);

    void terminate();

private:
    ITerminate&        terminate_;
    IGetTopValueFrame& get_top_value_frame_;
    IAddValue&         add_value_;
    IGetValueDelta&    value_delta_;
};

template<typename IT, typename IGTVF, typename IAV, typename IGVD>
dbuct_value_terminator<IT, IGTVF, IAV, IGVD>::dbuct_value_terminator(
        IT&    terminate,
        IGTVF& get_top_value_frame,
        IAV&   add_value,
        IGVD&  value_delta)
    : terminate_(terminate)
    , get_top_value_frame_(get_top_value_frame)
    , add_value_(add_value)
    , value_delta_(value_delta)
{}

template<typename IT, typename IGTVF, typename IAV, typename IGVD>
void dbuct_value_terminator<IT, IGTVF, IAV, IGVD>::terminate()
{
    terminate_.terminate();
    add_value_.add_value(value_delta_.get_value_delta(get_top_value_frame_.top().handle));
}

}

#endif
