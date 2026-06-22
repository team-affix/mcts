#ifndef EDGE_MAP_TABLE_HPP
#define EDGE_MAP_TABLE_HPP

#include <cstdint>
#include <unordered_map>
#include <utility>

namespace monte_carlo
{

// Concrete edge-visit counter using a sorted map.
// Stores the number of times each directed (parent, child) edge was traversed
// during the selection phase of MCTS.
//
// Template parameters:
//   NodeHandle — node identifier type (must be comparable/orderable for Map)
//   Map        — map template, e.g. std::map
//                Key type will be std::pair<NodeHandle, NodeHandle>
//
// Fulfils IGetEdgeVisits / ISetEdgeVisits policy requirements for sim:
//   get_edge_visits(const NodeHandle& parent, const NodeHandle& child) -> size_t
//   set_edge_visits(const NodeHandle& parent, const NodeHandle& child, size_t) -> void
//
// Zero-default contract: returns 0 for any edge never written.

template<
    typename NodeHandle,
    template<typename...> typename Map
>
struct edge_map_table
{
    size_t get_edge_visits(const NodeHandle& parent, const NodeHandle& child) const;
    void   set_edge_visits(const NodeHandle& parent, const NodeHandle& child, size_t v);

private:
    Map<std::pair<NodeHandle, NodeHandle>, size_t> edges_;
};

template<typename NodeHandle, template<typename...> typename Map>
size_t edge_map_table<NodeHandle, Map>::get_edge_visits(
    const NodeHandle& parent, const NodeHandle& child) const
{
    auto it = edges_.find({parent, child});
    return it == edges_.end() ? 0 : it->second;
}

template<typename NodeHandle, template<typename...> typename Map>
void edge_map_table<NodeHandle, Map>::set_edge_visits(
    const NodeHandle& parent, const NodeHandle& child, size_t v)
{
    edges_[{parent, child}] = v;
}

// ---------------------------------------------------------------------------
// Fast variant for integer NodeHandles.
// Packs (parent, child) into a single uint64 key, allowing std::unordered_map
// without any custom hash specialisation.
// ---------------------------------------------------------------------------

template<typename IntNodeHandle = int>
struct int_edge_unordered_table
{
    size_t get_edge_visits(const IntNodeHandle& parent, const IntNodeHandle& child) const;
    void   set_edge_visits(const IntNodeHandle& parent, const IntNodeHandle& child, size_t v);

private:
    static constexpr uint64_t pack(IntNodeHandle p, IntNodeHandle c)
    {
        return (static_cast<uint64_t>(static_cast<uint32_t>(p)) << 32)
             |  static_cast<uint64_t>(static_cast<uint32_t>(c));
    }

    std::unordered_map<uint64_t, size_t> edges_;
};

template<typename IntNodeHandle>
size_t int_edge_unordered_table<IntNodeHandle>::get_edge_visits(
    const IntNodeHandle& parent, const IntNodeHandle& child) const
{
    auto it = edges_.find(pack(parent, child));
    return it == edges_.end() ? 0 : it->second;
}

template<typename IntNodeHandle>
void int_edge_unordered_table<IntNodeHandle>::set_edge_visits(
    const IntNodeHandle& parent, const IntNodeHandle& child, size_t v)
{
    edges_[pack(parent, child)] = v;
}

} // namespace monte_carlo

#endif // EDGE_MAP_TABLE_HPP
