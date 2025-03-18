# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    """
    The bridge crossing analysis.
    For the bridge grid:
    - Discount needs to be high enough to make the far reward (10) worth pursuing
    - Noise needs to be low enough to make crossing the bridge safe enough
    """
    answerDiscount = 0.9  # High discount to value future rewards
    answerNoise = 0.01    # Low noise to reduce chance of falling off bridge
    return answerDiscount, answerNoise

def question3a():
    """
    Question 3a: Prefer the close reward with probability 1
    
    We want the agent to:
    1. Prefer the close reward (1) over the distant reward (10)
    2. Avoid the negative rewards (-10)
    3. Take the shortest path to the close reward
    
    This means:
    - Low discount to prefer immediate rewards
    - Low noise for predictable movement
    - Small negative living reward to encourage quick paths
    """
    answerDiscount = 0.3    # Low discount to prefer closer rewards
    answerNoise = 0.0      # No noise for predictable movement
    answerLivingReward = -1.0  # Small penalty for living to encourage quick paths
    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    Question 3b: Prefer the distant reward with probability 1
    
    We want the agent to:
    1. Prefer the distant reward (10) over the closer reward (1)
    2. Avoid the negative rewards (-10)
    3. Take a longer path to get the better reward
    4. Go south at (2,3) and (2,4) to avoid risky paths
    
    This means:
    - High discount to value the distant reward
    - High noise to make edge paths very risky
    - Zero living reward as path length doesn't matter
    """
    answerDiscount = 0.4    # Lower discount to reduce attraction to immediate rewards
    answerNoise = 0.3      # Higher noise to make edge paths risky
    answerLivingReward = -2.0  # Larger negative reward to force safer paths
    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    Question 3c: Prefer the distant reward and risk the cliff
    
    We want the agent to:
    1. Prefer the distant reward (10) over the closer reward (1)
    2. Take the risky path near the cliff
    3. Go east along bottom, then up to the distant reward
    
    This means:
    - High discount to value the distant reward
    - Low noise to make risky path viable
    - High negative living reward to encourage shortest path
    """
    answerDiscount = 0.9      # High discount to value the distant reward
    answerNoise = 0.0        # No noise to make risky path viable
    answerLivingReward = -0.5  # Negative reward to encourage shortest path
    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    Question 3d: Prefer the distant reward with a safer path
    
    We want the agent to:
    1. Prefer the distant reward (10) over the closer reward (1)
    2. Take a safer path avoiding the cliff
    3. Go north from start and eventually reach (4,2)
    
    This means:
    - High discount to value the distant reward
    - Some noise to make cliff paths risky
    - Small positive living reward to encourage exploration
    """
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    Question 3e: Avoid all terminal states
    
    We want the agent to:
    1. Avoid all terminal states (both positive and negative)
    2. Prefer living to any terminal state
    3. Stay away from risky edges
    
    This means:
    - Very low discount to make future rewards irrelevant
    - High noise to make terminal states risky
    - Very high positive living reward to make living better than any terminal state
    """
    answerDiscount = 0.1
    answerNoise = 0.3
    answerLivingReward = 11.0
    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    Question 6: Optimal Learning Rate and Epsilon
    
    Is it possible to get optimal policy with >99% probability in 50 iterations?
    
    Analysis:
    1. With epsilon = 0, no exploration means can't reliably find optimal policy
    2. Even with exploration, 50 iterations is too few for >99% confidence
    3. No combination of epsilon and learning rate can guarantee this
    """
    return 'NOT POSSIBLE'

if __name__ == '__main__':
    print 'Answers to analysis questions:'
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print '  Question %s:\t%s' % (q, str(response))
