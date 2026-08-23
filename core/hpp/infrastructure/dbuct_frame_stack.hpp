#ifndef DBUCT_FRAME_STACK_HPP
#define DBUCT_FRAME_STACK_HPP

#include <cstddef>
#include <deque>
#include <stack>
#include "value_objects/dbuct_frame.hpp"

namespace monte_carlo
{

template<typename INodeHandle>
struct dbuct_frame_stack
{
    dbuct_frame_stack(dbuct_frame<INodeHandle> root);

    void                      push(const dbuct_frame<INodeHandle>& f);
    void                      pop();
    dbuct_frame<INodeHandle>& top();
    size_t                    size() const;

private:
    using stack_t = std::stack<dbuct_frame<INodeHandle>>;

    stack_t frames_;
};

template<typename INodeHandle>
dbuct_frame_stack<INodeHandle>::dbuct_frame_stack(dbuct_frame<INodeHandle> root)
    : frames_(std::deque<dbuct_frame<INodeHandle>>{root})
{}

template<typename INodeHandle>
void dbuct_frame_stack<INodeHandle>::push(const dbuct_frame<INodeHandle>& f)
{
    frames_.push(f);
}

template<typename INodeHandle>
void dbuct_frame_stack<INodeHandle>::pop()
{
    frames_.pop();
}

template<typename INodeHandle>
dbuct_frame<INodeHandle>& dbuct_frame_stack<INodeHandle>::top()
{
    return frames_.top();
}

template<typename INodeHandle>
size_t dbuct_frame_stack<INodeHandle>::size() const
{
    return frames_.size();
}

}

#endif
