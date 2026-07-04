#ifndef MAP_TABLE_HPP
#define MAP_TABLE_HPP

#include <cstddef>
#include <map>
#include <unordered_map>

namespace monte_carlo
{

// visits_table<NodeHandle, Map>
//
// Per-node visit counter, keyed by NodeHandle.
//
// Satisfies:
//   IGetVisits: get_visits(const NodeHandle&) -> size_t  (0 if unseen)
//   ISetVisits: set_visits(const NodeHandle&, size_t) -> void
//
// Map parameter:
//   visits_table<int, std::map>           — ordered
//   visits_table<int, std::unordered_map> — hash map, requires std::hash<NodeHandle>

template<
    typename NodeHandle,
    template<typename...> typename Map
>
struct visits_table
{
    size_t get_visits(const NodeHandle& h) const;
    void   set_visits(const NodeHandle& h, size_t v);

private:
    Map<NodeHandle, size_t> visits_;
};

// value_table<NodeHandle, IFloat, Map>
//
// Per-node accumulated reward, keyed by NodeHandle.
//
// Satisfies:
//   IGetValue: get_value(const NodeHandle&) -> IFloat  (IFloat{} if unseen)
//   ISetValue: set_value(const NodeHandle&, IFloat) -> void
//
// Map parameter:
//   value_table<int, double, std::map>           — ordered
//   value_table<int, double, std::unordered_map> — hash map

template<
    typename NodeHandle,
    typename IFloat,
    template<typename...> typename Map
>
struct value_table
{
    IFloat get_value(const NodeHandle& h) const;
    void   set_value(const NodeHandle& h, IFloat v);

private:
    Map<NodeHandle, IFloat> values_;
};

// ---------------------------------------------------------------------------
// visits_table member function definitions
// ---------------------------------------------------------------------------

template<typename NodeHandle, template<typename...> typename Map>
size_t visits_table<NodeHandle, Map>::get_visits(const NodeHandle& h) const
{
    auto it = visits_.find(h);
    if (it == visits_.end()) return 0;
    return it->second;
}

template<typename NodeHandle, template<typename...> typename Map>
void visits_table<NodeHandle, Map>::set_visits(const NodeHandle& h, size_t v)
{
    visits_[h] = v;
}

// ---------------------------------------------------------------------------
// value_table member function definitions
// ---------------------------------------------------------------------------

template<typename NodeHandle, typename IFloat, template<typename...> typename Map>
IFloat value_table<NodeHandle, IFloat, Map>::get_value(const NodeHandle& h) const
{
    auto it = values_.find(h);
    if (it == values_.end()) return IFloat{};
    return it->second;
}

template<typename NodeHandle, typename IFloat, template<typename...> typename Map>
void value_table<NodeHandle, IFloat, Map>::set_value(const NodeHandle& h, IFloat v)
{
    values_[h] = v;
}

} // namespace monte_carlo

#endif // MAP_TABLE_HPP
