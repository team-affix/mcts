#ifndef RANDOM_ROLLOUT_HPP
#define RANDOM_ROLLOUT_HPP

#include <random>
#include <cstddef>

namespace monte_carlo
{

// random_rollout<IChoice, IRndGen, IGetChoiceCount, IGetChoiceAt>
//
// Concrete IRolloutChoose implementation: picks a choice uniformly at random.
//
// Standard parameter order: domain type first, then policies.
//   IChoice         — the choice value type returned
//   IRndGen         — random engine (e.g. std::mt19937)
//   IGetChoiceCount — get_choice_count() -> size_t
//   IGetChoiceAt    — get_choice_at.at(size_t) -> IChoice

template<
    typename IChoice,
    typename IRndGen,
    typename IGetChoiceCount,
    typename IGetChoiceAt
>
struct random_rollout
{
    random_rollout(IRndGen& rnd_gen);

    IChoice rollout_choose(IGetChoiceCount& get_choice_count, IGetChoiceAt& get_choice_at);

private:
    IRndGen& rnd_gen_;
};

// ---------------------------------------------------------------------------
// member function definitions
// ---------------------------------------------------------------------------

template<typename IChoice, typename IRndGen, typename IGetChoiceCount, typename IGetChoiceAt>
random_rollout<IChoice, IRndGen, IGetChoiceCount, IGetChoiceAt>::random_rollout(IRndGen& rnd_gen)
    : rnd_gen_(rnd_gen)
{}

template<typename IChoice, typename IRndGen, typename IGetChoiceCount, typename IGetChoiceAt>
IChoice random_rollout<IChoice, IRndGen, IGetChoiceCount, IGetChoiceAt>::rollout_choose(
    IGetChoiceCount& get_choice_count,
    IGetChoiceAt&    get_choice_at)
{
    std::uniform_int_distribution<size_t> dist(0, get_choice_count.size() - 1);
    return get_choice_at.at(dist(rnd_gen_));
}

} // namespace monte_carlo

#endif // RANDOM_ROLLOUT_HPP
