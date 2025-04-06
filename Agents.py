from ShortCutEnvironment import *


class Agent:
    def __init__(self, environment, epsilon=0.1, alpha=0.1, gamma=1.0, n_steps: int = 1, seed: int = None) -> None:
        """
        Initializer method for Q-Learning agent model, sets the values necessary to complete all other methods.

        :param environment: Agent environment.
        :param epsilon: Exploration constant.
        :param alpha: Learning constant.
        :param gamma: Decay constant.
        :param n_steps: Number of steps the agent extends out to.
        :param seed: Randomness seed.
        """

        self.env: Environment = environment
        self.n_states: int = self.env.state_size()
        self.n_actions: int = self.env.action_size()
        self.epsilon: float = epsilon
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.n_steps: int = n_steps
        self.Q: np.ndarray = np.zeros((self.n_states, self.n_actions))

        if seed is not None:
            np.random.seed(seed)  # Used to ensure randomness, but can be filled to give repeatability in training.

    def select_action(self, state: int) -> int | np.ndarray[int]:
        """
        Epsilon greedy policy action selector, randomly generates a probability and correspondingly return an action.

        :param state: Current state.

        :return: Selected action.
        """

        # Choose random action
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        # Choose best action
        return np.argmax(self.Q[state])

    def update(self, **args) -> None:
        """
        Update the expected reward values (Q) table utilizing a formula.
        """

        raise NotImplementedError("Should include an update function.")

    def train(self, n_episodes) -> np.array:
        """
        Trains the agent by running multiple episodes, returning the cumulative rewards across episodes as an array.

        :param n_episodes: Number of training episodes.

        :return: Array of cumulative rewards.
        """

        raise NotImplementedError("Should include a training function.")


class QLearningAgent(Agent):

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Update the expected reward values (Q) table utilizing the formula:
        Q(s,a) = Q(s, a) + alf(r + g*max_a(Q(s')) - Q(s, a)), using Q table, alpha, and gamma.

        :param state: Current state.
        :param action: Selected action.
        :param reward: Reward of applying selected action in current state.
        :param next_state: New state after applying selected action in current state.
        """

        # Update the expected reward for selected state and action
        update = reward + self.gamma * np.amax(self.Q[next_state]) - self.Q[state, action]
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
            state = self.env.reset()
            done = self.env.done()
            cumulative_reward = 0

            # Run episode
            while not done:
                # Select and complete an action, and cumulate reward
                action = self.select_action(state)
                reward = self.env.step(action)
                cumulative_reward += reward
                next_state = self.env.state()
                done = self.env.done()
                # Update expected reward if necessary
                if not done:
                    self.update(state, action, reward, next_state)
                # Move along to the next state
                state = next_state

            # Record episode cumulative reward
            episode_returns[e] = cumulative_reward

        return episode_returns


class SARSAAgent(Agent):

    def update(self, state: int, action: int, reward: float, next_state: int, next_action: int) -> None:
        """
        Update the expected reward values (Q) table utilizing the formula:
        Q(s,a) = Q(s, a) + alf(r + g*Q(s',a') - Q(s, a)), using Q table, alpha, and gamma.

        :param state: Current state.
        :param action: Selected action.
        :param reward: Reward of applying selected action in current state.
        :param next_state: New state after applying selected action in current state.
        :param next_action: Selected optimal action for next state.
        """

        # Update the expected reward for selected state and action
        update = reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action]
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
            state = self.env.reset()
            action = self.select_action(state)
            done = self.env.done()
            cumulative_reward = 0

            # Run episode
            while not done:
                # Select and complete an action, and cumulate reward
                reward = self.env.step(action)
                cumulative_reward += reward
                next_state = self.env.state()
                next_action = self.select_action(next_state)
                done = self.env.done()
                # Update expected reward if necessary
                if not done:
                    self.update(state, action, reward, next_state, next_action)
                # Move along to the next state
                state = next_state
                action = next_action

            # Record episode cumulative reward
            episode_returns[e] = cumulative_reward

        return episode_returns


class ExpectedSARSAAgent(Agent):

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Update the expected reward values (Q) table utilizing the formula:
        Q(s,a) = Q(s, a) + alf(r + g * [sum(policy(p)) for all p * Q(s',a)] - Q(s, a)), using Q table, alpha, and gamma.

        :param state: Current state.
        :param action: Selected action.
        :param reward: Reward of applying selected action in current state.
        :param next_state: New state after applying selected action in current state.
        """
        # Calculates action probabilities
        action_probabilities = np.full(self.n_actions, self.epsilon / self.n_actions)
        best_action = self.select_action(next_state)
        action_probabilities[best_action] += 1 - self.epsilon

        # Sums Expected SARSA values
        expected_SARSA = sum(action_probabilities[act] * self.Q[next_state, act] for act in range(self.Q.shape[1]))

        # Update the expected reward for selected state and action
        update = reward + self.gamma * expected_SARSA - self.Q[state, action]
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
            state = self.env.reset()
            done = self.env.done()
            cumulative_reward = 0

            # Run episode
            while not done:
                # Select and complete an action, and cumulate reward
                action = self.select_action(state)
                reward = self.env.step(action)
                cumulative_reward += reward
                next_state = self.env.state()
                done = self.env.done()
                # Update expected reward if necessary
                if not done:
                    self.update(state, action, reward, next_state)
                # Move along to the next state
                state = next_state

            # Record episode cumulative reward
            episode_returns[e] = cumulative_reward

        return episode_returns


class nStepSARSAAgent(Agent):

    def update(self, state: int, action: int, reward: np.array, update_state: int, update_action: int) -> None:
        """
        Update the expected reward values (Q) table utilizing the formula:
        Q(s',a') = Q(s, a) + a((r_t+1 + g * r_t+2 + ... + g^n-1 r_t+n Q(s,a)) for n_steps - Q(s, a)),
        using Q table, alpha, and gamma.

        :param state: Current state.
        :param action: Selected action.
        :param reward: Reward of applying optimal actions in past states.
        :param update_state: State being updated in the Q-table.
        :param update_action: Action being update in the Q-table.
        """

        # Calculate n_step_return
        n_step_return = sum(self.gamma ** step * reward[step] for step in range(len(reward)))
        n_step_return += self.gamma ** len(reward) * self.Q[state, action]

        # Update the expected reward for selected state and action
        update = n_step_return - self.Q[update_state, update_action]
        self.Q[update_state, update_action] = self.Q[update_state, update_action] + self.alpha * update

    def train(self, n_episodes: int) -> np.array:
        """
        Trains the agent by running multiple episodes, returning the cumulative rewards across episodes as an array.

        :param n_episodes: Number of training episodes.

        :return: Array of cumulative rewards.
        """

        episode_returns = np.zeros(n_episodes)
        for e in range(n_episodes):
            # Reset the episodic variables
            current_state = self.env.reset()
            update_state = current_state
            current_action = self.select_action(current_state)
            update_action = current_action
            done = self.env.done()
            cumulative_reward = 0
            n_step_rewards = np.zeros(self.n_steps)
            previous_states = []

            # Takes n (+1 above) actions to generate n_step_rewards
            for step in range(self.n_steps):
                previous_states.append(current_state)
                n_step_rewards[step] = self.env.step(current_action)
                current_state = self.env.state()
                current_action = self.select_action(current_state)
                done = self.env.done()

                # Breaks if terminal state is reached
                if done:
                    break

            # Run episode
            while not done:

                # Updating past rewards and accumulating the reward
                n_step_rewards = np.roll(n_step_rewards, -1)
                n_step_rewards[-1] = self.env.step(current_action)
                cumulative_reward += n_step_rewards[-1]
                done = self.env.done()

                # Makes sure terminal state hasn't been reached
                if not done:
                    # Go to the next state and action
                    current_state = self.env.state()
                    current_action = self.select_action(current_state)

                    # Update expected reward
                    self.update(current_state, current_action, n_step_rewards, update_state, update_action)

                    # Takes next state being updated from previous state list
                    update_state = previous_states.pop(0)
                    update_action = self.select_action(update_state)
                    previous_states.append(current_state)

            # Sums non-summed rewards
            cumulative_reward += sum(n_step_rewards)

            # Removes T-1 state from previous state list, setting it as the cap for the update function
            cap_state = previous_states.pop(-1)
            cap_action = self.select_action(cap_state)

            # Removes terminal reward (0) for rewards list
            n_step_rewards = n_step_rewards[:-1]

            # Updates non-updated states
            for update_state in previous_states:

                # Removes the oldest reward from reward list and selects action
                n_step_rewards = n_step_rewards[1:]
                update_action = self.select_action(update_state)

                # Update expected reward
                self.update(cap_state, cap_action, n_step_rewards, update_state, update_action)

            # Record episode cumulative reward
            episode_returns[e] = cumulative_reward

        return episode_returns
