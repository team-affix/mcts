#ifndef HIGHEST_MEAN_CHILD_FINDER_HPP
#define HIGHEST_MEAN_CHILD_FINDER_HPP

#include <cstddef>
#include <optional>

namespace monte_carlo
{

// highest_mean_child_finder
//
// Finds the choice leading to the visited child of a node with the highest
// mean value (its value over its visits), the first of them on a tie, or
// nothing while no child has a visit.

template<
    typename INodeHandle,
    typename IChoice,
    typename IFloat,
    typename IGetVisits,
    typename IGetValue,
    typename IWalker,
    typename IGetChoiceCount,
    typename IGetChoiceAt
>
struct highest_mean_child_finder
{
    highest_mean_child_finder(IGetVisits& get_visits,
                              IGetValue&  get_value,
                              IWalker&    walker);

    std::optional<IChoice> find(const INodeHandle&     node,
                                const IGetChoiceCount& get_choice_count,
                                const IGetChoiceAt&    get_choice_at) const;

private:
    IGetVisits& get_visits_;
    IGetValue&  get_value_;
    IWalker&    walker_;
};

template<typename INH, typename IC, typename IF, typename IGVis, typename IGVal, typename IW,
         typename IGCC, typename IGCA>
highest_mean_child_finder<INH, IC, IF, IGVis, IGVal, IW, IGCC, IGCA>::highest_mean_child_finder(
        IGVis& get_visits,
        IGVal& get_value,
        IW&    walker)
    : get_visits_(get_visits)
    , get_value_(get_value)
    , walker_(walker)
{}

template<typename INH, typename IC, typename IF, typename IGVis, typename IGVal, typename IW,
         typename IGCC, typename IGCA>
std::optional<IC> highest_mean_child_finder<INH, IC, IF, IGVis, IGVal, IW, IGCC, IGCA>::find(
        const INH&  node,
        const IGCC& get_choice_count,
        const IGCA& get_choice_at) const
{
    std::optional<IC> best;
    IF                best_mean = 0;
    for (size_t index = 0; index < get_choice_count.size(); ++index)
    {
        const IC     choice       = get_choice_at.at(index);
        const INH    child        = walker_.walk(node, choice);
        const size_t child_visits = get_visits_.get_visits(child);
        if (child_visits == 0)
            continue;
        const IF   mean   = get_value_.get_value(child) / static_cast<IF>(child_visits);
        const bool better = !best.has_value() || mean > best_mean;
        if (!better)
            continue;
        best      = choice;
        best_mean = mean;
    }
    return best;
}

}

#endif
