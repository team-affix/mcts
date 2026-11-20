#ifndef BEST_PATH_RECORDING_TERMINATOR_HPP
#define BEST_PATH_RECORDING_TERMINATOR_HPP

namespace monte_carlo
{

// best_path_recording_terminator
//
// Ends an episode as the terminator it wraps does (crediting the path and
// ending the rollout), then records the episode's path if it is the best so
// far (best_path_recorder).

template<
    typename ITerminate,
    typename IRecord
>
struct best_path_recording_terminator
{
    best_path_recording_terminator(ITerminate& terminator,
                                   IRecord&    recorder);

    void terminate();

private:
    ITerminate& terminator_;
    IRecord&    recorder_;
};

template<typename IT, typename IR>
best_path_recording_terminator<IT, IR>::best_path_recording_terminator(
        IT& terminator,
        IR& recorder)
    : terminator_(terminator)
    , recorder_(recorder)
{}

template<typename IT, typename IR>
void best_path_recording_terminator<IT, IR>::terminate()
{
    terminator_.terminate();
    recorder_.record();
}

}

#endif
