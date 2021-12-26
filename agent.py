import numpy as np


class ActorCriticAgent:
    def __init__(self, n_states, n_actions, gamma) -> None:
        self.actor = self.Actor(n_states, n_actions)
        self.critic = self.Critic(n_states, n_actions, gamma)
        self.gamma = gamma

    def take_action(self, state):
        return self.actor.take_action(state)

    def learn(self, state, action, reward, next_state, alpha):
        td_error = self.critic.get_td_error(
            state, action, reward, next_state, alpha)
        self.actor.update_policy(state, action, td_error, alpha)
        self.critic.update_V(state, td_error, alpha)

    class Actor:
        def __init__(self, n_states, n_actions) -> None:
            self.n_actions = n_actions
            self.Q = np.random.uniform(size=(n_states, n_actions))
            self.policy = np.full((n_states, n_actions), 1/n_actions)

        def __softmax(self, x):
            return np.exp(x)/np.sum(np.exp(x), axis=0)

        def update_policy(self, state, action, td_error, alpha):
            self.__update_Q(state, action, td_error, alpha)
            self.policy[state] = self.__softmax(self.Q[state])

        def take_action(self, state):
            actions = [i for i in range(self.n_actions)]
            action = np.random.choice(actions, p=self.policy[state])
            return action

        def __update_Q(self, state, action, td_error, alpha):
            self.Q[state][action] += alpha*td_error

    class Critic:
        def __init__(self, n_states, n_actions, gamma) -> None:
            self.n_actions = n_actions
            self.V = np.zeros((n_states))
            self.gamma = gamma

        def get_td_error(self, state, action, reward, next_state, alpha):
            gain = reward+self.gamma*self.V[next_state]
            estimated = self.V[state]
            td_error = gain-estimated
            return td_error

        def update_V(self, state, td_error, alpha):
            self.V[state] += alpha*td_error
