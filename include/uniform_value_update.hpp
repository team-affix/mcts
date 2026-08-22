#ifndef UNIFORM_VALUE_UPDATE_HPP
#define UNIFORM_VALUE_UPDATE_HPP

namespace monte_carlo
{

template<
    typename INodeHandle,
    typename IGetValue,
    typename ISetValue,
    typename IGetValueDelta
>
struct uniform_value_update
{
    uniform_value_update(IGetValue&      get_value,
                         ISetValue&      set_value,
                         IGetValueDelta& value_delta);

    void update(const INodeHandle& node);

private:
    IGetValue&      get_value_;
    ISetValue&      set_value_;
    IGetValueDelta& value_delta_;
};

template<typename INodeHandle, typename IGVal, typename ISVal, typename IGVD>
uniform_value_update<INodeHandle, IGVal, ISVal, IGVD>::uniform_value_update(
        IGVal& get_value,
        ISVal& set_value,
        IGVD&  value_delta)
    : get_value_(get_value)
    , set_value_(set_value)
    , value_delta_(value_delta)
{}

template<typename INodeHandle, typename IGVal, typename ISVal, typename IGVD>
void uniform_value_update<INodeHandle, IGVal, ISVal, IGVD>::update(const INodeHandle& node)
{
    set_value_.set_value(node, get_value_.get_value(node) + value_delta_.get_value_delta(node));
}

}

#endif
