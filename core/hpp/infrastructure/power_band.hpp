#ifndef POWER_BAND_HPP
#define POWER_BAND_HPP

#include <cmath>
#include <cstddef>

namespace monte_carlo
{

// power_band
//
// The width of the band within which a child's normalized mean counts as tied
// with the best at a node: A * N^(-r), for a node with N visits. A is the
// starting width on the normalized [0, 1] scale; r > 0 is how fast the band
// shrinks as the node gathers visits, so any real difference eventually lies
// outside it.

template<typename IFloat>
struct power_band
{
    power_band(IFloat width, IFloat rate);

    IFloat get_band(size_t parent_visits) const;

private:
    IFloat width_;
    IFloat rate_;
};

template<typename IFloat>
power_band<IFloat>::power_band(IFloat width, IFloat rate)
    : width_(width)
    , rate_(rate)
{}

template<typename IFloat>
IFloat power_band<IFloat>::get_band(size_t parent_visits) const
{
    return width_ * std::pow(static_cast<IFloat>(parent_visits), -rate_);
}

}

#endif
