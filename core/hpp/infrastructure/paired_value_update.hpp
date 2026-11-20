#ifndef PAIRED_VALUE_UPDATE_HPP
#define PAIRED_VALUE_UPDATE_HPP

namespace monte_carlo
{

// paired_value_update
//
// Updates a node with two value updates, the first then the second. Use when a
// creditor takes one update but two statistics are kept per node, as mixmax
// keeps both the reward sum and the best reward.

template<
    typename INodeHandle,
    typename IUpdateNode0,
    typename IUpdateNode1
>
struct paired_value_update
{
    paired_value_update(IUpdateNode0& update_node0,
                        IUpdateNode1& update_node1);

    void update(const INodeHandle& node);

private:
    IUpdateNode0& update_node0_;
    IUpdateNode1& update_node1_;
};

template<typename INodeHandle, typename IUN0, typename IUN1>
paired_value_update<INodeHandle, IUN0, IUN1>::paired_value_update(
        IUN0& update_node0,
        IUN1& update_node1)
    : update_node0_(update_node0)
    , update_node1_(update_node1)
{}

template<typename INodeHandle, typename IUN0, typename IUN1>
void paired_value_update<INodeHandle, IUN0, IUN1>::update(const INodeHandle& node)
{
    update_node0_.update(node);
    update_node1_.update(node);
}

}

#endif
