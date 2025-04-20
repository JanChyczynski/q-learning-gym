import math
from random import random
from typing import Tuple

import gym
import time

class QLearner:
    def __init__(self, learning_rate, discount_factor, experiment_rate, discretization_buckets):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.experiment_rate = experiment_rate
        self.discretization_buckets = discretization_buckets
        self.default_q = 0
        self.environment = gym.make("CartPole-v1", render_mode="human")
        self.attempt_no = 1
        self.upper_bounds = [
            self.environment.observation_space.high[0],
            0.5,
            self.environment.observation_space.high[2],
            math.radians(50)
        ]
        self.lower_bounds = [
            self.environment.observation_space.low[0],
            -0.5,
            self.environment.observation_space.low[2],
            -math.radians(50)
        ]
        self.q_dict = {}

    def learn(self, max_attempts):
        for _ in range(max_attempts):
            reward_sum = self.attempt()
            print(reward_sum)

    def attempt(self):
        observation = self.discretise(self.environment.reset()[0])
        done = False
        reward_sum = 0.0
        while not done:
            # self.environment.render()
            action = self.pick_action(observation)
            new_observation, reward, done, truncated, info = self.environment.step(action)

            new_observation = self.discretise(new_observation)
            self.update_knowledge(action, observation, new_observation, reward)
            observation = new_observation
            reward_sum += reward
        self.attempt_no += 1
        return reward_sum

    def discretise(self, observation):
        return [round((val - low)/(up-low) * (self.discretization_buckets-1))
                for val, low, up in zip(observation, self.lower_bounds, self.upper_bounds)]

    def pick_action(self, observation):
        if random() > self.experiment_rate:
            best_action, _ = self.best_action(observation)
            return best_action
        else:
            return self.environment.action_space.sample()

    def best_action(self, observation):
        best_action = None
        best_q = self.default_q
        for action in range(self.environment.action_space.n):
            current_q = self.q_dict.get((tuple(observation), action), self.default_q)
            if best_action is None or current_q > best_q or (current_q == best_q and random() > 0.5):
                best_action = action
                best_q = current_q
        assert best_action is not None
        return best_action, best_q

    def update_knowledge(self, action, observation, new_observation, reward):
        new_q = reward + self.discount_factor * self.best_action(new_observation)[1]
        if (tuple(observation), action) in self.q_dict.keys():
            old_q = self.q_dict.get((tuple(observation), action), self.default_q)
            self.q_dict[(tuple(observation), action)] = (1-self.learning_rate) * old_q + self.learning_rate * new_q
        else:
            self.q_dict[(tuple(observation), action)] = new_q

def main():
    learner = QLearner(0.5, 0.5, 0.5, 4)
    learner.learn(10000)


if __name__ == '__main__':
    main()
