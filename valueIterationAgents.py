# -*- coding: utf-8 -*-
# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        states = self.mdp.getStates()
        
        # For each iteration
        for i in range(self.iterations):
            # Create a new counter for updated values
            next_values = util.Counter()
            
            # For each state
            for state in states:
                # Skip terminal states as they have value 0
                if not self.mdp.isTerminal(state):
                    # Get possible actions for this state
                    actions = self.mdp.getPossibleActions(state)
                    # If there are actions available
                    if actions:
                        # Find maximum Q-value among all actions
                        max_q = float("-inf")
                        for action in actions:
                            q_value = self.computeQValueFromValues(state, action)
                            max_q = max(max_q, q_value)
                        next_values[state] = max_q
            
            # Update values for next iteration
            self.values = next_values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # Initialize Q-value
        q_value = 0
        
        # Get all possible next states and their probabilities
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        
        # For each possible next state
        for nextState, prob in transitions:
            # Get the reward for this transition
            reward = self.mdp.getReward(state, action, nextState)
            # Add to Q-value using Bellman equation:
            # Q(s,a) = sum( P(s'|s,a) * [R(s,a,s') + gamma * V(s')] )
            q_value += prob * (reward + self.discount * self.values[nextState])
            
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # Get list of legal actions for this state
        legal_actions = self.mdp.getPossibleActions(state)
        
        # If no legal actions, return None (terminal state)
        if not legal_actions:
            return None
            
        # Find action with maximum Q-value
        max_value = float("-inf")
        best_action = None
        
        # For each legal action, compute its Q-value and keep track of the best
        for action in legal_actions:
            q_value = self.computeQValueFromValues(state, action)
            if q_value > max_value:
                max_value = q_value
                best_action = action
                
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
