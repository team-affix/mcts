#ifndef DBUCT_VISIT_CREDITOR_HPP
#define DBUCT_VISIT_CREDITOR_HPP

namespace monte_carlo
{

template<typename IAddVisits>
struct dbuct_visit_creditor
{
    dbuct_visit_creditor(IAddVisits& visit_adder);

    void credit();

private:
    IAddVisits& visit_adder_;
};

template<typename IAV>
dbuct_visit_creditor<IAV>::dbuct_visit_creditor(IAV& visit_adder)
    : visit_adder_(visit_adder)
{}

template<typename IAV>
void dbuct_visit_creditor<IAV>::credit()
{
    visit_adder_.add_visits(1);
}

}

#endif
