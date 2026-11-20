#ifndef BEST_PATH_CHILD_FINDER_HPP
#define BEST_PATH_CHILD_FINDER_HPP

#include <optional>

namespace monte_carlo
{

// best_path_child_finder
//
// Finds the choice the best episode so far made at a node, or nothing if that
// episode did not pass through it.

template<
    typename INodeHandle,
    typename IChoice,
    typename IFindStep,
    typename IGetChoiceCount,
    typename IGetChoiceAt
>
struct best_path_child_finder
{
    best_path_child_finder(IFindStep& find_step);

    std::optional<IChoice> find(const INodeHandle&     node,
                                const IGetChoiceCount& get_choice_count,
                                const IGetChoiceAt&    get_choice_at) const;

private:
    IFindStep& find_step_;
};

template<typename INH, typename IC, typename IFS, typename IGCC, typename IGCA>
best_path_child_finder<INH, IC, IFS, IGCC, IGCA>::best_path_child_finder(IFS& find_step)
    : find_step_(find_step)
{}

template<typename INH, typename IC, typename IFS, typename IGCC, typename IGCA>
std::optional<IC> best_path_child_finder<INH, IC, IFS, IGCC, IGCA>::find(
        const INH&  node,
        const IGCC&,
        const IGCA&) const
{
    return find_step_.find_step(node);
}

}

#endif
