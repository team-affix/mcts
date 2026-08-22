#ifndef DBUCT_FRAME_HPP
#define DBUCT_FRAME_HPP

#include <compare>
#include <cstddef>

namespace monte_carlo
{

template<typename INodeHandle>
struct dbuct_frame
{
    dbuct_frame(INodeHandle handle, size_t budget);

    INodeHandle handle;
    size_t      budget;
    size_t      visit_lump;

    auto operator<=>(const dbuct_frame&) const = default;
};

template<typename INodeHandle>
dbuct_frame<INodeHandle>::dbuct_frame(INodeHandle handle, size_t budget)
    : handle(handle)
    , budget(budget)
    , visit_lump(0)
{}

}

#endif
