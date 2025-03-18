import numpy as np
from ShortCutEnvironment import env


class QLearningAgent(object):

    def __init__(self, n_actions: int, n_states: int, epsilon=0.1, alpha=0.1, gamma=1.0, seed: int = None) -> None:
        """
        Initializer method for this agent model, sets the values necessary to complete all other methods.

        :param n_actions: Number of actions an agent can take at any state.
        :param n_states: Number of states the environment contains.
        :param epsilon: Exploration constant.
        :param alpha: Learning constant.
        :param gamma: Decay constant.
        :param seed: Randomness seed.
        """

        self.n_actions: int = n_actions
        self.n_states: int = n_states
        self.epsilon: float = epsilon
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.Q: np.ndarray = np.zeros((n_states, n_actions))

        np.random.seed(seed)  # Used to ensure randomness, but can be filled to give repeatability in training.

    def select_action(self, state: int) -> int:
        """
        Epsilon greedy policy action selector, randomly generates a probability and correspondingly return an action.

        :param state: Current state.

        :return: Selected action.
        """

        # Choose random action
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        # Choose best action
        return np.argmax(self.Q[state])[0]

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Update the expected reward values (Q) table utilizing the formula:
        Q(s,a) = Q(s, a) + a(r + g*max_a(Q(s')) - Q(s, a)), using Q table, alpha, and gamma.

        :param state: Current state.
        :param action: Selected action.
        :param reward: Reward of applying selected action in current state.
        :param next_state: New state after applying selected action in current state.
        """

        # Update the expected reward for selected state and action
        update = reward + self.gamma * np.argmax(self.Q[next_state]) - self.Q[state, action]
        self.Q[state, action] = self.Q[state, action] + self.alpha * update

    def train(self, n_episodes: int) -> np.array:
        """
        Trains the agent by running multiple episodes, returning the cumulative rewards across episodes as an array.

        :param n_episodes: Number of training episodes.

        :return: Array of cumulative rewards.
        """

        episode_returns = np.zeros(n_episodes)
        for e in range(n_episodes):
            # Reset the episodic variables
            state = env.reset()
            done = env.done()
            cumulative_reward = 0

            # Run episode
            while not done:
                # Select and complete an action
                action = self.select_action(state)
                reward = env.step(action)
                next_state = env.state()
                done = env.done()
                # Update expected reward if necessary
                if not done:
                    self.update(state, action, reward, next_state)
                # Move along to the next state and cumulate reward
                state = next_state
                cumulative_reward += reward  # Should we add the reward of goal state?

            # Record episode cumulative reward
            episode_returns[e] = cumulative_reward

        return episode_returns


class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary

    def select_action(self, state):
        # TO DO: Implement policy
        action = None
        return action

    def update(self, state, action, reward, done):  # Augment arguments if necessary
        # TO DO: Implement SARSA update
        pass

    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes.
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        return episode_returns


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary

    def select_action(self, state):
        # TO DO: Implement policy
        action = None
        return action

    def update(self, state, action, reward, done):  # Augment arguments if necessary
        # TO DO: Implement Expected SARSA update
        pass

    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes.
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        return episode_returns


class nStepSARSAAgent(object):

    def __init__(self, n_actions, n_states, n, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary

    def select_action(self, state):
        # TO DO: Implement policy
        action = None
        return action

    def update(self, states, actions, rewards, done):  # Augment arguments if necessary
        # TO DO: Implement n-step SARSA update
        pass

    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes.
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        return episode_returns
