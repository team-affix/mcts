#ifndef MAP_TABLE_HPP
#define MAP_TABLE_HPP

#include <cstddef>
#include <map>
#include <unordered_map>

namespace monte_carlo
{

// map_table<NodeHandle, IFloat, Map>
//
// Flat stats store keyed by NodeHandle. Replaces tree_node — there is no tree
// node concept; this is simply a map from node identity to {visits, value}.
//
// Satisfies all four stat accessor interfaces expected by sim:
//   IGetVisits:  get_visits(const NodeHandle&) -> size_t   (0 if unseen — zero-default contract)
//   IGetValue:   get_value(const NodeHandle&)  -> IFloat    (0 if unseen — zero-default contract)
//   ISetVisits:  set_visits(const NodeHandle&, size_t)
//   ISetValue:   set_value(const NodeHandle&, IFloat)
//
// Pass the same map_table object for all four sim constructor parameters.
//
// Map parameter — supply explicitly, no default:
//   map_table<int, double, std::map>           — ordered, no hash required
//   map_table<int, double, std::unordered_map> — hash map, requires std::hash<NodeHandle>

template<
    typename NodeHandle,
    typename IFloat,
    template<typename...> typename Map
>
struct map_table
{
    size_t get_visits(const NodeHandle& node_handle) const;
    IFloat get_value(const NodeHandle&  node_handle) const;
    void   set_visits(const NodeHandle& node_handle, size_t v);
    void   set_value(const NodeHandle&  node_handle, IFloat v);

private:
    struct node_stats { IFloat value{}; size_t visits{}; };
    Map<NodeHandle, node_stats> nodes_;
};

// ---------------------------------------------------------------------------
// member function definitions
// ---------------------------------------------------------------------------

template<typename NodeHandle, typename IFloat, template<typename...> typename Map>
size_t map_table<NodeHandle, IFloat, Map>::get_visits(const NodeHandle& node_handle) const
{
    auto it = nodes_.find(node_handle);
    if (it == nodes_.end()) return 0;
    return it->second.visits;
}

template<typename NodeHandle, typename IFloat, template<typename...> typename Map>
IFloat map_table<NodeHandle, IFloat, Map>::get_value(const NodeHandle& node_handle) const
{
    auto it = nodes_.find(node_handle);
    if (it == nodes_.end()) return IFloat{};
    return it->second.value;
}

template<typename NodeHandle, typename IFloat, template<typename...> typename Map>
void map_table<NodeHandle, IFloat, Map>::set_visits(const NodeHandle& node_handle, size_t v)
{
    nodes_[node_handle].visits = v;
}

template<typename NodeHandle, typename IFloat, template<typename...> typename Map>
void map_table<NodeHandle, IFloat, Map>::set_value(const NodeHandle& node_handle, IFloat v)
{
    nodes_[node_handle].value = v;
}

} // namespace monte_carlo

#endif // MAP_TABLE_HPP
