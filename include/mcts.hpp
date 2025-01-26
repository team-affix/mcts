#ifndef MCTS_HPP
#define MCTS_HPP

#include <math.h>
#include <map>
#include <functional>
#include <random>

namespace monte_carlo
{
    struct tree_node
    {
        double m_value;
        size_t m_visits;
        std::map<size_t, tree_node> m_children;
    };

    template<typename RND_DEVICE_T>
    double tree_search(
        tree_node&                           a_node,
        const std::function<size_t(size_t)>& a_act_fxn,
        const std::function<double()>&       a_value_fxn,
        size_t                               a_remaining_action_count,
        const double                         a_exploration_constant,
        RND_DEVICE_T&                        a_rnd_dev)
    {
        // performs a series of finalizing moves until we hit terminal state
        const auto& l_rollout = [&a_act_fxn, &a_value_fxn, a_remaining_action_count, &a_rnd_dev]
        {
            // helper alias (see below usage of `urd`)
            using urd = std::uniform_int_distribution<size_t>;

            // create local register of remaining action count
            size_t l_remaining_action_count = a_remaining_action_count;

            // execute simulation until we reach a terminal state
            while (l_remaining_action_count > 0)
                l_remaining_action_count = a_act_fxn(urd(0, l_remaining_action_count-1)(a_rnd_dev));

            // get the value of this terminal state, and return it.
            return a_value_fxn();
        };

        // utilize monte-carlo tree search with UCB1 heuristic
        const auto& l_UCB1 = [&a_node, a_exploration_constant] (const tree_node& a_child)
        {
            if (a_child.m_visits == 0) return std::numeric_limits<double>::infinity();
            // compute the node's exploitative value
            double l_child_exploitative_value = a_child.m_value / (double)a_child.m_visits;
            // compute the node's explorative value
            double l_child_explorative_value = sqrt(log(a_node.m_visits) / (double)a_child.m_visits);
            // ucb1 = exploitative + c * explorative
            return l_child_exploitative_value + a_exploration_constant * l_child_explorative_value;
        };

        //////////////////////////////////////////////////////////////////
        ///////////////////////// CHECK LEAF NODE ////////////////////////
        //////////////////////////////////////////////////////////////////

        if (a_node.m_visits++ == 0) return a_node.m_value = l_rollout();

        //////////////////////////////////////////////////////////////////
        /////////////////////// CHECK TERMINAL STATE /////////////////////
        //////////////////////////////////////////////////////////////////

        if (a_remaining_action_count == 0)
        {
            // get the value for this terminal state
            double l_val = a_value_fxn();
            // add this value to the current aggregate value
            a_node.m_value += l_val;
            // return the increment in value
            return l_val;
        }

        //////////////////////////////////////////////////////////////////
        //////////////////// SELECT / CHOOSE ACTION //////////////////////
        //////////////////////////////////////////////////////////////////

        double l_best_UCB1_score = -INFINITY;
        size_t l_selected_action;

        // get child which maximizes UCB1
        for (size_t i = 0; i < a_remaining_action_count; ++i)
        {
            double l_UCB1_score = l_UCB1(a_node.m_children[i]);
            if (l_UCB1_score <= l_best_UCB1_score) continue;
            l_best_UCB1_score = l_UCB1_score;
            l_selected_action = i;
        }

        //////////////////////////////////////////////////////////////////
        ///////////////////////// EXECUTE ACTION /////////////////////////
        //////////////////////////////////////////////////////////////////

        size_t l_child_remaining_actions = a_act_fxn(l_selected_action);

        //////////////////////////////////////////////////////////////////
        ////////////////////////////// RECUR /////////////////////////////
        //////////////////////////////////////////////////////////////////

        // execute mcts on the chosen child
        double l_result = tree_search(a_node.m_children[l_selected_action], a_act_fxn, a_value_fxn, l_child_remaining_actions, a_exploration_constant, a_rnd_dev);
        // add the result to the current node's value
        a_node.m_value += l_result;
        // return the simulation result so the root of the tree can see how this simulation went
        return l_result;
        
    }
}

#endif
