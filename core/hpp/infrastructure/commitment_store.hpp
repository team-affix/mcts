#ifndef COMMITMENT_STORE_HPP
#define COMMITMENT_STORE_HPP

#include <cstddef>
#include <optional>

namespace monte_carlo
{

// commitment_store
//
// The choices the search has committed to, by node, and the frontier: the node
// below the last commitment (the root before any), where the next commitment
// will be decided, with its visit count when it became the frontier.

template<
    typename INodeHandle,
    typename IChoice,
    template<typename...> typename Map
>
struct commitment_store
{
    commitment_store(INodeHandle root);

    std::optional<IChoice> find_commitment(const INodeHandle& node) const;
    void                   commit(const INodeHandle& node, const IChoice& choice,
                                  const INodeHandle& child, size_t child_visits);
    INodeHandle            frontier() const;
    size_t                 frontier_base_visits() const;
    size_t                 commitment_count() const;

private:
    Map<INodeHandle, IChoice> commitments_;
    INodeHandle               frontier_;
    size_t                    frontier_base_visits_;
};

template<typename INH, typename IC, template<typename...> typename Map>
commitment_store<INH, IC, Map>::commitment_store(INH root)
    : commitments_()
    , frontier_(root)
    , frontier_base_visits_(0)
{}

template<typename INH, typename IC, template<typename...> typename Map>
std::optional<IC> commitment_store<INH, IC, Map>::find_commitment(const INH& node) const
{
    const auto found = commitments_.find(node);
    if (found == commitments_.end())
        return std::nullopt;
    return found->second;
}

template<typename INH, typename IC, template<typename...> typename Map>
void commitment_store<INH, IC, Map>::commit(const INH& node, const IC& choice,
                                            const INH& child, size_t child_visits)
{
    commitments_[node]    = choice;
    frontier_             = child;
    frontier_base_visits_ = child_visits;
}

template<typename INH, typename IC, template<typename...> typename Map>
INH commitment_store<INH, IC, Map>::frontier() const
{
    return frontier_;
}

template<typename INH, typename IC, template<typename...> typename Map>
size_t commitment_store<INH, IC, Map>::frontier_base_visits() const
{
    return frontier_base_visits_;
}

template<typename INH, typename IC, template<typename...> typename Map>
size_t commitment_store<INH, IC, Map>::commitment_count() const
{
    return commitments_.size();
}

}

#endif
