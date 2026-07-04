#ifndef DISPATCHES_TABLE_HPP
#define DISPATCHES_TABLE_HPP

#include <cstddef>
#include <map>
#include <unordered_map>

namespace monte_carlo
{

// dispatches_table<NodeHandle, Map>
//
// Per-node dispatch counter, keyed by NodeHandle.
//
// Satisfies the two dispatch accessor interfaces expected by dbuct:
//   IGetDispatches: get_dispatches(const NodeHandle&) -> size_t  (0 if unseen)
//   ISetDispatches: set_dispatches(const NodeHandle&, size_t)
//
// Pass the same dispatches_table object for both dbuct dispatch parameters.
//
// Map parameter:
//   dispatches_table<int, std::map>           — ordered, no hash required
//   dispatches_table<int, std::unordered_map> — hash map, requires std::hash<NodeHandle>

template<
    typename NodeHandle,
    template<typename...> typename Map
>
struct dispatches_table
{
    size_t get_dispatches(const NodeHandle& h) const;
    void   set_dispatches(const NodeHandle& h, size_t v);

private:
    Map<NodeHandle, size_t> counts_;
};

// ---------------------------------------------------------------------------
// member function definitions
// ---------------------------------------------------------------------------

template<typename NodeHandle, template<typename...> typename Map>
size_t dispatches_table<NodeHandle, Map>::get_dispatches(const NodeHandle& h) const
{
    auto it = counts_.find(h);
    if (it == counts_.end()) return 0;
    return it->second;
}

template<typename NodeHandle, template<typename...> typename Map>
void dispatches_table<NodeHandle, Map>::set_dispatches(const NodeHandle& h, size_t v)
{
    counts_[h] = v;
}

} // namespace monte_carlo

#endif // DISPATCHES_TABLE_HPP
