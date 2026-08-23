#ifndef UCT_BACKPROP_PATH_HPP
#define UCT_BACKPROP_PATH_HPP

#include <cstddef>
#include <stack>
#include <vector>

namespace monte_carlo
{

template<typename INodeHandle>
struct uct_backprop_path
{
    uct_backprop_path(INodeHandle root);

    void        push(const INodeHandle& node);
    void        pop();
    INodeHandle top() const;
    size_t      size() const;

private:
    using stack_t = std::stack<INodeHandle, std::vector<INodeHandle>>;

    stack_t nodes_;
};

template<typename INodeHandle>
uct_backprop_path<INodeHandle>::uct_backprop_path(INodeHandle root)
    : nodes_(std::vector<INodeHandle>{root})
{}

template<typename INodeHandle>
void uct_backprop_path<INodeHandle>::push(const INodeHandle& node)
{
    nodes_.push(node);
}

template<typename INodeHandle>
void uct_backprop_path<INodeHandle>::pop()
{
    nodes_.pop();
}

template<typename INodeHandle>
INodeHandle uct_backprop_path<INodeHandle>::top() const
{
    return nodes_.top();
}

template<typename INodeHandle>
size_t uct_backprop_path<INodeHandle>::size() const
{
    return nodes_.size();
}

}

#endif
