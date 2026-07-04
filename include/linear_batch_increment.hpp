#ifndef LINEAR_BATCH_INCREMENT_HPP
#define LINEAR_BATCH_INCREMENT_HPP

#include <cstddef>

namespace monte_carlo
{

// linear_batch_increment
//
// Concrete IComputeBatchSize policy.  Implements the formula:
//   compute_batch_size(D) = 1 + D / B
//
// where D is the pre-increment dispatch count supplied by the caller (dbuct)
// and B is the grant_increment_interval provided at construction.
//
// This is a pure, stateless computation; all dispatch counter management is
// left to dbuct via its IGetDispatches / ISetDispatches policy references.
//
// Special case: B = SIZE_MAX gives compute_batch_size(D) = 1 for all D,
// which is equivalent to vanilla UCT (one simulation per sub-budget).

struct linear_batch_increment
{
    explicit linear_batch_increment(size_t B);
    size_t compute_batch_size(size_t dispatch_count) const;

private:
    size_t B_;
};

// ---------------------------------------------------------------------------
// member function definitions
// ---------------------------------------------------------------------------

inline linear_batch_increment::linear_batch_increment(size_t B)
    : B_(B)
{}

inline size_t linear_batch_increment::compute_batch_size(size_t dispatch_count) const
{
    return 1 + dispatch_count / B_;
}

} // namespace monte_carlo

#endif // LINEAR_BATCH_INCREMENT_HPP
