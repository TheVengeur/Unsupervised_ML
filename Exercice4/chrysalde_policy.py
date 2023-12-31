from Exercice4.agent import Agent
from numpy.random import random_integers as rand

PROB_STAY_ON_REWARD =   80  # 80% stay in place when on reward
PROB_MOVE_TOWARDS_CLOSEST_EDGE =    20 # 20% move to closest edge, which has less chances of holding rewards

def chrysalde_policy(agent: Agent) -> str:
    """
    Policy of the agent
    return "left", "right", or "none"
    """
    actions = ["left", "right", "none"]

    BOARD_LEN = len(agent.known_rewards)
    AGENT_ON_LEFT_SIDE = agent.position <= BOARD_LEN / 2

    if agent.known_rewards[agent.position] != 0:
        if rand(0, 100) <= PROB_STAY_ON_REWARD:
            action = "none"
        elif rand(0, 100) <= PROB_MOVE_TOWARDS_CLOSEST_EDGE:
            action = "left" if AGENT_ON_LEFT_SIDE else "right"
        else:
            action = "right" if AGENT_ON_LEFT_SIDE else "left"
    else:
        if rand(0, 100) <= PROB_MOVE_TOWARDS_CLOSEST_EDGE:
            action = "left" if AGENT_ON_LEFT_SIDE else "right"
        else:
            action = "right" if AGENT_ON_LEFT_SIDE else "left"

    assert action in actions
    return action
