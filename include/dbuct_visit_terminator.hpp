#ifndef DBUCT_VISIT_TERMINATOR_HPP
#define DBUCT_VISIT_TERMINATOR_HPP

namespace monte_carlo
{

template<typename IAddVisits>
struct dbuct_visit_terminator
{
    dbuct_visit_terminator(IAddVisits& add_visits);

    void terminate();

private:
    IAddVisits& add_visits_;
};

template<typename IAV>
dbuct_visit_terminator<IAV>::dbuct_visit_terminator(IAV& add_visits)
    : add_visits_(add_visits)
{}

template<typename IAV>
void dbuct_visit_terminator<IAV>::terminate()
{
    add_visits_.add_visits(1);
}

}

#endif
