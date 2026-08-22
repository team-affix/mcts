#ifndef SIM_CURSOR_HPP
#define SIM_CURSOR_HPP

namespace monte_carlo
{

template<typename INodeHandle>
struct sim_cursor
{
    sim_cursor(INodeHandle root);

    INodeHandle get_current_node() const;
    void        set_current_node(INodeHandle node);

private:
    INodeHandle current_node_;
};

template<typename INodeHandle>
sim_cursor<INodeHandle>::sim_cursor(INodeHandle root)
    : current_node_(root)
{}

template<typename INodeHandle>
INodeHandle sim_cursor<INodeHandle>::get_current_node() const
{
    return current_node_;
}

template<typename INodeHandle>
void sim_cursor<INodeHandle>::set_current_node(INodeHandle node)
{
    current_node_ = node;
}

}

#endif
