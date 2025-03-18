# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        
        # Initialize Q-values as a nested Counter
        self.qValues = util.Counter()  # A Counter is a dict with default 0

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # Return the Q-value for the state-action pair
        # Counter automatically returns 0.0 for never-seen state-action pairs
        return self.qValues[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        # Get legal actions in this state
        legalActions = self.getLegalActions(state)
        
        # If no legal actions (terminal state), return 0.0
        if not legalActions:
            return 0.0
            
        # Find maximum Q-value over all legal actions
        maxQ = float("-inf")
        for action in legalActions:
            q_value = self.getQValue(state, action)
            maxQ = max(maxQ, q_value)
            
        return maxQ

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # Get legal actions in this state
        legalActions = self.getLegalActions(state)
        
        # If no legal actions (terminal state), return None
        if not legalActions:
            return None
            
        # Find action with maximum Q-value
        maxQ = float("-inf")
        bestAction = None
        
        # Check each legal action
        for action in legalActions:
            q_value = self.getQValue(state, action)
            # Update best action if this Q-value is higher
            if q_value > maxQ:
                maxQ = q_value
                bestAction = action
            # Break ties randomly
            elif q_value == maxQ and random.random() < 0.5:
                bestAction = action
                
        return bestAction

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Get legal actions
        legalActions = self.getLegalActions(state)
        action = None
        
        # Return None if no legal actions
        if not legalActions:
            return action
        
        # Epsilon-greedy action selection
        if util.flipCoin(self.epsilon):
            # Random action
            action = random.choice(legalActions)
        else:
            # Best policy action
            action = self.getPolicy(state)
        
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here using the Q-learning update rule:
          Q(s,a) = Q(s,a) + alpha * [R + gamma * maxQ(s',a') - Q(s,a)]
          
          alpha = learning rate
          gamma = discount factor
          R = reward
          maxQ(s',a') = maximum Q-value for any action in the next state
        """
        # Get the current Q-value
        currentQ = self.getQValue(state, action)
        
        # Get the maximum Q-value for the next state
        nextMaxQ = self.computeValueFromQValues(nextState)
        
        # Calculate the temporal difference target
        # reward + (discount * next state's max Q-value)
        target = reward + (self.discount * nextMaxQ)
        
        # Update the Q-value using the learning rate (alpha)
        # Q(s,a) = Q(s,a) + alpha * [target - Q(s,a)]
        newQ = currentQ + self.alpha * (target - currentQ)
        
        # Store the new Q-value
        self.qValues[(state, action)] = newQ

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        return features * self.weights

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # Get the feature vector for current state-action pair
        features = self.featExtractor.getFeatures(state, action)
        
        # Get the current Q-value
        currentQ = self.getQValue(state, action)
        
        # Get the maximum Q-value for the next state
        nextMaxQ = self.computeValueFromQValues(nextState)
        
        # Calculate the temporal difference target
        # reward + (discount * next state's max Q-value)
        target = reward + (self.discount * nextMaxQ)
        
        # Calculate the difference between target and current Q-value
        difference = target - currentQ
        
        # Update each weight using the learning rate and feature value
        for feature in features:
            self.weights[feature] += self.alpha * difference * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
