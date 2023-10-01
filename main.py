import random
import matplotlib.pyplot as plt
import numpy as np

DYNAMICS = {
    # Keys = (initial state, action, next state)
    # Values = Probability
    (1, 1, 4): 1.0,
    (1, 2, 4): 1.0,
    (2, 1, 4): .8, (2, 1, 5): .2,
    (2, 2, 4): .6, (2, 2, 5): .4,
    (3, 1, 4): .9, (3, 1, 5): .1,
    (3, 2, 5): 1.0,
    (4, 1, 6): 1.0,
    (4, 2, 6): .3, (4, 2, 7): .7,
    (5, 1, 6): .3, (5, 1, 7): .7,
    (5, 2, 7): 1.0,
}

REWARDS = {
    # Keys = States
    # Values = Tuples of rewards of taking actions sorted by index
    1: [7, 10],
    2: [-3, 5],
    3: [4, -6],
    4: [9, -1],
    5: [-8, 2],
}

pi = {
    # Keys = States
    # Values = Tuples of probabillities of taking actions sorted by index
    1: (.5, .5),
    2: (.7, .3),
    3: (.9, .1),
    4: (.4, .6),
    5: (.2, .8)
}
START_STATES = [1, 2, 3]
ACTION_SPACE = [1, 2]
STATE_SPACE = [1, 2, 3, 4, 5]
START_STATE_DISTRIBUTION = [.6, .3, .1]

def runEpisode(policy: dict, gamma: float) -> float:
    # Pull state from start state distribution
    state = random.choices(START_STATES, weights=START_STATE_DISTRIBUTION, k=1)[0]
    discounted_return = 0
    # Range is 2 since we know the episode will terminate in 2 time steps
    for t in range(2):
        action = random.choices(ACTION_SPACE, weights=list(policy.get(state)), k=1)[0]
        next_state = random.choices(get_next_states(state, action), weights=get_next_state_probabilities(state, action), k=1)[0]
        reward = (gamma**t) * REWARDS.get(state)[action-1]
        state = next_state
        discounted_return += reward
    return discounted_return

def get_next_states(initial_state: int, action: int, dynamics: dict = DYNAMICS): 
    # Gets the possible next states
    return [key[2] for key in dynamics if key[0] == initial_state and key[1] == action]

def get_next_state_probabilities(initial_state: int, action: int, dynamics: dict= DYNAMICS): 
    # Gets the probabilities of the possible next states
    return [dynamics[key] for key in dynamics if key[0] == initial_state and key[1] == action]

def approximate_objective_returns(n_episodes: int, policy: dict, gamma: float) -> float:
    total_discounted_return = 0
    avg_discounted_returns = []
    discounted_returns = []

    for i in range(n_episodes):
        discounted_return = runEpisode(policy, gamma)
        total_discounted_return += discounted_return
        avg_discounted_returns.append(total_discounted_return/(i+1))
        discounted_returns.append(discounted_return)
    return [avg_discounted_returns, discounted_returns]

# PROBLEM 2d)
def policy_gen_eval(n_episodes: int, gamma: float):
    best_policy = None
    best_policy_performance = -np.inf
    best_performances = []
    performances = []
    for i in range(n_episodes):
        policy = {}
        for s in STATE_SPACE:
            action_probabilities = [0 for _ in range(len(ACTION_SPACE))]
            # Among the possible values, make it determinisitc
            action_probabilities[random.choice(ACTION_SPACE)-1] = 1
            policy[s] = action_probabilities

        # Average the performance over 100 Episodes per the instructions
        performance = np.mean([runEpisode(policy=policy, gamma=gamma) for _ in range(100)])

        if performance > best_policy_performance:
            best_policy_performance = performance
            best_policy = policy

        best_performances.append(best_policy_performance)
    return [best_policy, best_performances, performances]


if __name__ == "__main__":
    # n_episodes = 150000
    # # PROBLEM 2b)
    # returns = approximate_objective_returns(n_episodes = 150000, policy=pi, gamma=.9)
    # print(f'Final averaged objective function: {returns[0][len(returns[0])-1]}')
    # print(f'Variance of objective function: {np.var(returns[1])}')


    # PROBLEM 2c)
    # for gamma in [.25, .5, .75, .99]:
    #     expected_value = 5.22 + gamma*2.7106
    #     actual_value = approximate_objective_returns(n_episodes = 150000, policy=pi, gamma=gamma)[len(avg_returns)-1]
    #     error = np.abs(expected_value - actual_value)
    #     print(expected_value, actual_value, error)

    # PROBLEM 2a)
    # Plotting
    # plt.plot(range(n_episodes), avg_returns)
    # plt.xlabel('Episode Number')
    # plt.ylabel('Average Discounted Return')
    # plt.title('Approximate Objective Function Across Episodes')
    # plt.grid(True)
    # plt.show()

    # PROBLEM 2db)
    n_episodes = 250
    best_performances = policy_gen_eval(n_episodes=n_episodes, gamma=.9)[1]
    # Plotting
    plt.plot(range(n_episodes), best_performances)
    plt.xticks(np.arange(0, n_episodes+1, 10))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Discounted Return')
    plt.title('Approximate Objective Function Across Episodes Using Policy Optimization')
    plt.grid(True)
    plt.show()