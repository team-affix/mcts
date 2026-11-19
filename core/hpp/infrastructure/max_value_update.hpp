#ifndef MAX_VALUE_UPDATE_HPP
#define MAX_VALUE_UPDATE_HPP

namespace monte_carlo
{

// max_value_update
//
// Single-player value update: a node's value is the best episode reward ever
// backed up through it, not the sum. On the node's first visit the reward is
// taken as is, since an unseen node reads as IFloat{} and rewards may be
// negative. The first visit is told apart by the visit count, which must
// already include the current episode (uct_value_creditor credits the visit
// before it updates the value).

template<
    typename INodeHandle,
    typename IGetVisits,
    typename IGetValue,
    typename ISetValue,
    typename IGetValueDelta
>
struct max_value_update
{
    max_value_update(IGetVisits&     get_visits,
                     IGetValue&      get_value,
                     ISetValue&      set_value,
                     IGetValueDelta& value_delta);

    void update(const INodeHandle& node);

private:
    IGetVisits&     get_visits_;
    IGetValue&      get_value_;
    ISetValue&      set_value_;
    IGetValueDelta& value_delta_;
};

template<typename INodeHandle, typename IGVis, typename IGVal, typename ISVal, typename IGVD>
max_value_update<INodeHandle, IGVis, IGVal, ISVal, IGVD>::max_value_update(
        IGVis& get_visits,
        IGVal& get_value,
        ISVal& set_value,
        IGVD&  value_delta)
    : get_visits_(get_visits)
    , get_value_(get_value)
    , set_value_(set_value)
    , value_delta_(value_delta)
{}

template<typename INodeHandle, typename IGVis, typename IGVal, typename ISVal, typename IGVD>
void max_value_update<INodeHandle, IGVis, IGVal, ISVal, IGVD>::update(const INodeHandle& node)
{
    const auto reward      = value_delta_.get_value_delta(node);
    const bool first_visit = get_visits_.get_visits(node) <= 1;

    if (!first_visit && get_value_.get_value(node) >= reward)
        return;

    set_value_.set_value(node, reward);
}

}

#endif
