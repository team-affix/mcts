#ifndef VALUE_BOUNDS_HPP
#define VALUE_BOUNDS_HPP

#include <limits>

namespace monte_carlo
{

// value_bounds
//
// The lowest and highest node values seen so far, used to put values on a
// [0, 1] scale. Until two different values have been seen there is no spread,
// and every value normalizes to 0.

template<typename IFloat>
struct value_bounds
{
    value_bounds();

    void   include(IFloat value);
    IFloat normalize(IFloat value) const;

private:
    IFloat lowest_;
    IFloat highest_;
};

template<typename IFloat>
value_bounds<IFloat>::value_bounds()
    : lowest_(std::numeric_limits<IFloat>::infinity())
    , highest_(-std::numeric_limits<IFloat>::infinity())
{}

template<typename IFloat>
void value_bounds<IFloat>::include(IFloat value)
{
    if (value < lowest_)
        lowest_ = value;
    if (value > highest_)
        highest_ = value;
}

template<typename IFloat>
IFloat value_bounds<IFloat>::normalize(IFloat value) const
{
    const bool spread = highest_ > lowest_;
    if (!spread)
        return 0;
    return (value - lowest_) / (highest_ - lowest_);
}

}

#endif
