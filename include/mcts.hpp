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

    template<typename ACTION_T, typename CONTEXT_T, typename RND_GEN_T>
    double tree_search(
        tree_node<ACTION_T>& a_node,
        CONTEXT_T&           a_context,
        const double         a_exploration_constant,
        RND_GEN_T&           a_rnd_gen)
    {
        // performs a series of finalizing moves until we hit terminal state
        const auto& l_rollout = [&a_context, &a_rnd_gen]
        {
            // helper alias (see below usage of `urd`)
            using urd = std::uniform_int_distribution<int>;

            int l_action_count;

            // execute simulation until we reach a terminal state
            while ((l_action_count = a_context.action_count()) > 0)
                a_context.act(
                    a_context.action(
                        urd(0, l_action_count-1)(a_rnd_gen)
                        )
                    );

            // get the value of this terminal state, and return it.
            return a_context.value();
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

        if (a_context.action_count() == 0)
        {
            // get the value for this terminal state
            double l_val = a_context.value();
            // add this value to the current aggregate value
            a_node.m_value += l_val;
            // return the increment in value
            return l_val;
        }

        //////////////////////////////////////////////////////////////////
        //////////////////// SELECT / CHOOSE ACTION //////////////////////
        //////////////////////////////////////////////////////////////////

        double   l_best_UCB1_score = -INFINITY;
        ACTION_T l_selected_action{};

        // get child which maximizes UCB1
        for (int i = 0; i < a_context.action_count(); ++i)
        {
            ACTION_T l_action = a_context.action(i);
            double l_UCB1_score = l_UCB1(a_node.m_children[l_action]);
            if (l_UCB1_score <= l_best_UCB1_score) continue;
            l_best_UCB1_score = l_UCB1_score;
            l_selected_action = l_action;
        }

        //////////////////////////////////////////////////////////////////
        ///////////////////////// EXECUTE ACTION /////////////////////////
        //////////////////////////////////////////////////////////////////

        a_context.act(l_selected_action);

        //////////////////////////////////////////////////////////////////
        ////////////////////////////// RECUR /////////////////////////////
        //////////////////////////////////////////////////////////////////

        // execute mcts on the chosen child
        double l_result = tree_search(a_node.m_children[l_selected_action], a_context, a_exploration_constant, a_rnd_gen);
        // add the result to the current node's value
        a_node.m_value += l_result;
        // return the simulation result so the root of the tree can see how this simulation went
        return l_result;
        
    }
}

#endif
