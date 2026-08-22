#ifndef DBUCT_ADD_VALUE_HPP
#define DBUCT_ADD_VALUE_HPP

#include "dbuct_value_frame.hpp"

namespace monte_carlo
{

template<
    typename INodeHandle,
    typename IFloat,
    typename IGetTopValueFrame,
    typename IGetValue,
    typename ISetValue
>
struct dbuct_add_value
{
    dbuct_add_value(IGetTopValueFrame& get_top_value_frame,
                    IGetValue&         get_value,
                    ISetValue&         set_value);

    void add_value(IFloat l);

private:
    IGetTopValueFrame& get_top_value_frame_;
    IGetValue&         get_value_;
    ISetValue&         set_value_;
};

template<typename INH, typename IF, typename IGTVF, typename IGVal, typename ISVal>
dbuct_add_value<INH, IF, IGTVF, IGVal, ISVal>::dbuct_add_value(
        IGTVF& get_top_value_frame,
        IGVal& get_value,
        ISVal& set_value)
    : get_top_value_frame_(get_top_value_frame)
    , get_value_(get_value)
    , set_value_(set_value)
{}

template<typename INH, typename IF, typename IGTVF, typename IGVal, typename ISVal>
void dbuct_add_value<INH, IF, IGTVF, IGVal, ISVal>::add_value(IF l)
{
    dbuct_value_frame<INH, IF>& f = get_top_value_frame_.top();
    set_value_.set_value(f.handle, get_value_.get_value(f.handle) + l);
    f.value_lump += l;
}

}

#endif
