#ifndef BEST_PATH_TERMINATOR_HPP
#define BEST_PATH_TERMINATOR_HPP

namespace monte_carlo
{

// best_path_terminator
//
// Ends an episode by grafting its rollout's path onto the tree when it earned
// the best reward so far (best_path_grafter), then terminating as UCT does
// (uct_terminator), which credits every node on the backprop path.

template<
    typename IGraft,
    typename ITerminate
>
struct best_path_terminator
{
    best_path_terminator(IGraft&     grafter,
                         ITerminate& terminator);

    void terminate();

private:
    IGraft&     grafter_;
    ITerminate& terminator_;
};

template<typename IG, typename IT>
best_path_terminator<IG, IT>::best_path_terminator(
        IG& grafter,
        IT& terminator)
    : grafter_(grafter)
    , terminator_(terminator)
{}

template<typename IG, typename IT>
void best_path_terminator<IG, IT>::terminate()
{
    grafter_.graft();
    terminator_.terminate();
}

}

#endif
