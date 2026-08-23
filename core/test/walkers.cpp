#include "walkers.hpp"

size_t VectorIntHash::operator()(const std::vector<int>& v) const noexcept
{
    size_t seed = v.size();
    for (int x : v)
        seed ^= static_cast<size_t>(x) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
    return seed;
}

int position_walker::walk(const int& node_handle, jump_t j) const
{
    return node_handle + j;
}

std::vector<int> path_walker::walk(const std::vector<int>& path, jump_t j) const
{
    std::vector<int> child = path;
    child.push_back(path.back() + j);
    return child;
}
