#ifndef WALKERS_HPP
#define WALKERS_HPP

#include <cstddef>
#include <unordered_map>
#include <vector>

// Shared test scaffolding for the track-navigation games used across the unit
// and integration suites.  A "jump" is a signed distance along the track.

using jump_t = int;

// Hash for std::vector<int>, enabling unordered_map keyed by path.
struct VectorIntHash
{
    size_t operator()(const std::vector<int>& v) const noexcept;
};

// Map alias used by the path-keyed banks.
template<typename K, typename V>
using path_unordered_map = std::unordered_map<K, V, VectorIntHash>;

// Walker for node-contraction tests: node handle is the current position.
struct position_walker
{
    int walk(const int& node_handle, jump_t j) const;
};

// Walker for tree-based tests: node handle is the full path of positions from
// root, making every distinct traversal route a unique node.
struct path_walker
{
    std::vector<int> walk(const std::vector<int>& path, jump_t j) const;
};

#endif
