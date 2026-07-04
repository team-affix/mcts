#ifndef VALUE_TABLE_HPP
#define VALUE_TABLE_HPP

#include <map>
#include <unordered_map>

namespace monte_carlo
{

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
// member function definitions
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

#endif // VALUE_TABLE_HPP
