#include "infrastructure/linear_batch_increment.hpp"

namespace monte_carlo
{

linear_batch_increment::linear_batch_increment(size_t B)
    : B_(B)
{}

size_t linear_batch_increment::compute_batch_size(size_t dispatch_count) const
{
    return 1 + dispatch_count / B_;
}

} // namespace monte_carlo
