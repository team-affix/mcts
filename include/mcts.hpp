#ifndef MCTS_HPP
#define MCTS_HPP

#include <math.h>
#include <map>
#include <functional>
#include <random>

namespace monte_carlo
{
    template<typename ACTION_T>
    struct tree_node
    {
        double m_value;
        size_t m_visits;
        std::map<ACTION_T, tree_node> m_children;
    };

    template<typename ACTION_T, typename ACTION_CONTAINER_T, typename RND_GEN_T>
    double tree_search(
        tree_node<ACTION_T>&                 a_node,
        const ACTION_CONTAINER_T&            a_actions,
        const std::function<void(ACTION_T)>& a_act_fxn,
        const std::function<double()>&       a_value_fxn,
        const double                         a_exploration_constant,
        RND_GEN_T&                           a_rnd_gen)
    {
        // performs a series of finalizing moves until we hit terminal state
        const auto& l_rollout = [&a_actions, &a_act_fxn, &a_value_fxn, &a_rnd_gen]
        {
            // helper alias (see below usage of `urd`)
            using urd = std::uniform_int_distribution<int>;

            // execute simulation until we reach a terminal state
            while (a_actions.size() > 0)
                a_act_fxn(a_actions[urd(0, a_actions.size()-1)(a_rnd_gen)]);

            // get the value of this terminal state, and return it.
            return a_value_fxn();
        };

        // utilize monte-carlo tree search with UCB1 heuristic
        const auto& l_UCB1 = [&a_node, a_exploration_constant] (const tree_node<ACTION_T>& a_child)
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

        if (a_actions.size() == 0)
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

        double   l_best_UCB1_score       = -INFINITY;
        ACTION_T l_best_action{};

        // get child which maximizes UCB1
        for (const ACTION_T& l_action : a_actions)
        {
            double l_UCB1_score = l_UCB1(a_node.m_children[l_action]);
            if (l_UCB1_score <= l_best_UCB1_score)
                continue;
            l_best_UCB1_score = l_UCB1_score;
            l_best_action     = l_action;
        }

        //////////////////////////////////////////////////////////////////
        ///////////////////////// EXECUTE ACTION /////////////////////////
        //////////////////////////////////////////////////////////////////

        a_act_fxn(l_best_action);

        //////////////////////////////////////////////////////////////////
        ////////////////////////////// RECUR /////////////////////////////
        //////////////////////////////////////////////////////////////////

        // execute mcts on the chosen child
        double l_result = tree_search(a_node.m_children[l_best_action], a_actions, a_act_fxn, a_value_fxn, a_exploration_constant, a_rnd_gen);
        // add the result to the current node's value
        a_node.m_value += l_result;
        // return the simulation result so the root of the tree can see how this simulation went
        return l_result;
        
    }
}

#endif
