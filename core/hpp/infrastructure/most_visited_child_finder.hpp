#ifndef MOST_VISITED_CHILD_FINDER_HPP
#define MOST_VISITED_CHILD_FINDER_HPP

#include <cstddef>
#include <optional>

namespace monte_carlo
{

// most_visited_child_finder
//
// Finds the choice leading to a node's most visited child, the first of them
// on a tie, or nothing while no child has a visit.

template<
    typename INodeHandle,
    typename IChoice,
    typename IGetVisits,
    typename IWalker,
    typename IGetChoiceCount,
    typename IGetChoiceAt
>
struct most_visited_child_finder
{
    most_visited_child_finder(IGetVisits& get_visits,
                              IWalker&    walker);

    std::optional<IChoice> find(const INodeHandle&     node,
                                const IGetChoiceCount& get_choice_count,
                                const IGetChoiceAt&    get_choice_at) const;

private:
    IGetVisits& get_visits_;
    IWalker&    walker_;
};

template<typename INH, typename IC, typename IGVis, typename IW, typename IGCC, typename IGCA>
most_visited_child_finder<INH, IC, IGVis, IW, IGCC, IGCA>::most_visited_child_finder(
        IGVis& get_visits,
        IW&    walker)
    : get_visits_(get_visits)
    , walker_(walker)
{}

template<typename INH, typename IC, typename IGVis, typename IW, typename IGCC, typename IGCA>
std::optional<IC> most_visited_child_finder<INH, IC, IGVis, IW, IGCC, IGCA>::find(
        const INH&  node,
        const IGCC& get_choice_count,
        const IGCA& get_choice_at) const
{
    std::optional<IC> best;
    size_t            best_visits = 0;
    for (size_t index = 0; index < get_choice_count.size(); ++index)
    {
        const IC     choice       = get_choice_at.at(index);
        const size_t child_visits = get_visits_.get_visits(walker_.walk(node, choice));
        if (child_visits <= best_visits)
            continue;
        best        = choice;
        best_visits = child_visits;
    }
    return best;
}

}

#endif
