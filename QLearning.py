#Dewey Schoenfelder
import numpy as np

class QLearning(object):
    def __init__(self, R, goal_state=5):
        self.Q = np.zeros(R.shape)  # initialize Q to zeros
        self.R = R  # reward matrix
        self.num_states = R.shape[0]
        self.gamma = 0.8
        self.goal_state = goal_state
        self.current_state = np.random.randint(0, self.num_states)

    def get_reward_from_environment(self, action):
        # returns reward based on action
        reward = self.R[self.current_state, action]  # R table gives the reward for an action
        return reward

    def train(self, num_training_episodes):
        # one episode is from random start state to goal state
        self.Q = np.zeros(self.R.shape)  # clear Q

        for i in range(num_training_episodes):
            self.current_state = np.random.randint(0, self.num_states)  # random start state

            while True:
                valid_action_found = False

                while not valid_action_found:
                    possible_action = np.random.randint(0, self.num_states)  # pick random next state
                    reward = self.get_reward_from_environment(possible_action)
                    if reward >= 0:
                        valid_action_found = True

                next_state = possible_action
                qmax_next_state = self.get_QMax(next_state)

                # Update Q-value
                self.Q[self.current_state, possible_action] = reward + (self.gamma * qmax_next_state)

                self.current_state = possible_action

                if self.current_state == self.goal_state:
                    break

            print(f"Finished episode {i}, restarting environment\n")

    def get_QMax(self, next_state):
        return np.max(self.Q[next_state])
