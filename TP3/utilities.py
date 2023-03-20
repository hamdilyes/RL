import numpy as np


def run_N_episodes(env, agent, N_episodes=1):
    for _ in range(N_episodes):
        done = False
        state, _ = env.reset()
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update(state, action, reward, terminated, next_state)
            done = terminated or truncated

            state = next_state
    return env, agent


def rewardseq_to_returns(reward_list: list[float], gamma: float) -> list[float]:
    """
    Turns a list of rewards into the list of returns
    """
    G = 0
    returns_list = []
    for r in reward_list[::-1]:
        G = r + gamma * G
        returns_list.append(G)
    return returns_list[::-1]


def to_arr(value_dict):
    res = np.zeros(7)
    for state in value_dict:
        res[state] = value_dict[state]

    return res
