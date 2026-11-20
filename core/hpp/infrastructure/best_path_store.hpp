#ifndef BEST_PATH_STORE_HPP
#define BEST_PATH_STORE_HPP

#include <limits>
#include <optional>

namespace monte_carlo
{

// best_path_store
//
// The path of the episode with the best reward so far: the choice it made at
// each node it passed through, and that reward (minus infinity before any
// path is recorded).

template<
    typename INodeHandle,
    typename IChoice,
    typename IFloat,
    template<typename...> typename Map
>
struct best_path_store
{
    best_path_store();

    std::optional<IChoice> find_step(const INodeHandle& node) const;
    IFloat                 best_reward() const;
    void                   start_path(IFloat reward);
    void                   add_step(const INodeHandle& node, const IChoice& choice);

private:
    Map<INodeHandle, IChoice> steps_;
    IFloat                    best_reward_;
};

template<typename INH, typename IC, typename IF, template<typename...> typename Map>
best_path_store<INH, IC, IF, Map>::best_path_store()
    : steps_()
    , best_reward_(-std::numeric_limits<IF>::infinity())
{}

template<typename INH, typename IC, typename IF, template<typename...> typename Map>
std::optional<IC> best_path_store<INH, IC, IF, Map>::find_step(const INH& node) const
{
    const auto found = steps_.find(node);
    if (found == steps_.end())
        return std::nullopt;
    return found->second;
}

template<typename INH, typename IC, typename IF, template<typename...> typename Map>
IF best_path_store<INH, IC, IF, Map>::best_reward() const
{
    return best_reward_;
}

template<typename INH, typename IC, typename IF, template<typename...> typename Map>
void best_path_store<INH, IC, IF, Map>::start_path(IF reward)
{
    steps_.clear();
    best_reward_ = reward;
}

template<typename INH, typename IC, typename IF, template<typename...> typename Map>
void best_path_store<INH, IC, IF, Map>::add_step(const INH& node, const IC& choice)
{
    steps_[node] = choice;
}

}

#endif
