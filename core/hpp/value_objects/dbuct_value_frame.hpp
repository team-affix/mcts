#ifndef DBUCT_VALUE_FRAME_HPP
#define DBUCT_VALUE_FRAME_HPP

#include <compare>

namespace monte_carlo
{

template<typename INodeHandle, typename IFloat>
struct dbuct_value_frame
{
    dbuct_value_frame(INodeHandle handle);

    INodeHandle handle;
    IFloat      value_lump;

    auto operator<=>(const dbuct_value_frame&) const = default;
};

template<typename INodeHandle, typename IFloat>
dbuct_value_frame<INodeHandle, IFloat>::dbuct_value_frame(INodeHandle handle)
    : handle(handle)
    , value_lump(0)
{}

}

#endif
