import numpy as np
import scipy
import matplotlib
import ShortCutEnvironment as environnment


class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary

        # Initializes Q-tables
        self.Q_values = np.zeros((self.n_states, self.n_actions))
        
    def select_action(self, state):
        # TO DO: Implement policy

        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.n_actions)

        return np.argmax(self.Q_values[state])
        
    def update(self, state, action, reward, done): # Augment arguments if necessary
        # TO DO: Implement Q-learning update

        if not done:
            # Updates Q-values based on Q-learning equation
            self.Q_values[state, action] = self.Q_values[state, action] + self.alpha * (reward + (self.gamma * np.argmax(self.Q_values[state+1])) - self.Q_values[state, action])
    
    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the cumulative reward (=return) per episode
        episode_returns = []
        for episode in range(n_episodes):
            state = environnment.env.reset()
            cumulative_reward = 0
            done = environnment.env.done()

            while not done:

                action = self.select_action(state)
                reward = environnment.env.step(action)
                cumulative_reward += reward
                next_state = environnment.env.state()
                done = environnment.env.done()
                self.update(state, action, reward, done)
                state = next_state

            episode_returns.append(cumulative_reward)


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
        
    def update(self, state, action, reward, done): # Augment arguments if necessary
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
        
    def update(self, state, action, reward, done): # Augment arguments if necessary
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
        
    def update(self, states, actions, rewards, done): # Augment arguments if necessary
        # TO DO: Implement n-step SARSA update
        pass
    
    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        return episode_returns  
    
    
    