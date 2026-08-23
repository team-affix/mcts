#ifndef UCT_CURSOR_HPP
#define UCT_CURSOR_HPP

namespace monte_carlo
{

template<typename INodeHandle>
struct uct_cursor
{
    uct_cursor(INodeHandle root);

    INodeHandle get_current_node() const;
    void        set_current_node(INodeHandle node);

private:
    INodeHandle current_node_;
};

template<typename INodeHandle>
uct_cursor<INodeHandle>::uct_cursor(INodeHandle root)
    : current_node_(root)
{}

template<typename INodeHandle>
INodeHandle uct_cursor<INodeHandle>::get_current_node() const
{
    return current_node_;
}

template<typename INodeHandle>
void uct_cursor<INodeHandle>::set_current_node(INodeHandle node)
{
    current_node_ = node;
}

}

#endif
