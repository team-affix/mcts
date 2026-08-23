#ifndef DBUCT_VALUE_STACK_HPP
#define DBUCT_VALUE_STACK_HPP

#include <deque>
#include <stack>
#include "value_objects/dbuct_value_frame.hpp"

namespace monte_carlo
{

template<typename INodeHandle, typename IFloat>
struct dbuct_value_stack
{
    dbuct_value_stack(dbuct_value_frame<INodeHandle, IFloat> root);

    void                                    push(const dbuct_value_frame<INodeHandle, IFloat>& f);
    void                                    pop();
    dbuct_value_frame<INodeHandle, IFloat>& top();

private:
    using stack_t = std::stack<dbuct_value_frame<INodeHandle, IFloat>>;

    stack_t value_frames_;
};

template<typename INodeHandle, typename IFloat>
dbuct_value_stack<INodeHandle, IFloat>::dbuct_value_stack(
        dbuct_value_frame<INodeHandle, IFloat> root)
    : value_frames_(std::deque<dbuct_value_frame<INodeHandle, IFloat>>{root})
{}

template<typename INodeHandle, typename IFloat>
void dbuct_value_stack<INodeHandle, IFloat>::push(
        const dbuct_value_frame<INodeHandle, IFloat>& f)
{
    value_frames_.push(f);
}

template<typename INodeHandle, typename IFloat>
void dbuct_value_stack<INodeHandle, IFloat>::pop()
{
    value_frames_.pop();
}

template<typename INodeHandle, typename IFloat>
dbuct_value_frame<INodeHandle, IFloat>& dbuct_value_stack<INodeHandle, IFloat>::top()
{
    return value_frames_.top();
}

}

#endif
