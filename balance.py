import math
from random import random
from typing import Tuple

import gym
import time

import numpy as np
import csv
import matplotlib.pyplot as plt

ITERATIONS = 3000
MOVING_AVG_WINDOW = 900


class QLearner:
    def __init__(self, learning_rate, discount_factor, experiment_rate, discretization_buckets):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.experiment_rate = experiment_rate
        self.discretization_buckets = discretization_buckets
        self.default_q = 0
        self.environment = gym.make("CartPole-v1")
        # self.environment = gym.make("CartPole-v1", render_mode="human")
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
        rewards = [0] * max_attempts
        for i in range(max_attempts):
            reward_sum = self.attempt()
            rewards[i] = reward_sum

        tail = rewards[-100:]
        avg_tail = np.mean(tail)
        std_tail = np.std(tail)
        print(f"Params: lr={self.learning_rate}, df={self.discount_factor},"
            f"er={self.experiment_rate}, b={self.discretization_buckets}",
            f"=> avg: {avg_tail:.2f}, std: {std_tail:.2f}")

        return rewards

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
        return [round((val - low) / (up - low) * (self.discretization_buckets - 1))
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
            self.q_dict[(tuple(observation), action)] = (1 - self.learning_rate) * old_q + self.learning_rate * new_q
        else:
            self.q_dict[(tuple(observation), action)] = new_q


def moving_average(arr, window_size):
    return np.convolve(arr, np.ones(window_size) / window_size, mode='valid')


def gen_data(experiments, learning_rate, discount_factor, experiment_rate, discretization_buckets):
    results = []
    for _ in range(experiments):
        learner = QLearner(learning_rate, discount_factor, experiment_rate, discretization_buckets)
        results.append(learner.learn(ITERATIONS))
    results = np.array(results).reshape(ITERATIONS, -1)
    avg = np.average(results, axis=1)
    std = np.std(results, axis=1)

    filename = f"data_lr{learning_rate}_df{discount_factor}_er{experiment_rate}_b{discretization_buckets}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["avg", "std"])
        for a, s in zip(avg, std):
            writer.writerow([a, s])


def read_data(learning_rate, discount_factor, experiment_rate, discretization_buckets):
    filename = f"data_lr{learning_rate}_df{discount_factor}_er{experiment_rate}_b{discretization_buckets}.csv"
    avg, std = [], []
    with open(filename, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            avg.append(float(row["avg"]))
            std.append(float(row["std"]))
    return filename, np.array(avg), np.array(std)


def plot_all(datasets):
    plt.figure(figsize=(12, 6))
    for label, avg, std in datasets:
        x = np.arange(len(avg))
        ma = moving_average(avg, MOVING_AVG_WINDOW)
        std_ma = moving_average(std, MOVING_AVG_WINDOW)
        x_ma = np.arange(len(ma))

        plt.plot(x_ma, ma, label=label, linewidth=2)
        plt.plot(x_ma, ma + std_ma, linestyle='--', linewidth=1, alpha=0.5)
        plt.plot(x_ma, ma - std_ma, linestyle='--', linewidth=1, alpha=0.5)

    plt.title("Średnie wyniki z korytarzem odchyleń")
    plt.xlabel("Iteracja")
    plt.ylabel("Średnia nagroda")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Zbiorczy wykres tylko średnich
    plt.figure(figsize=(10, 5))
    for label, avg, _ in datasets:
        ma = moving_average(avg, MOVING_AVG_WINDOW)
        x_ma = np.arange(len(ma))
        plt.plot(x_ma, ma, label=label, linewidth=2)

    plt.title("Zbiorczy wykres średnich")
    plt.xlabel("Iteracja")
    plt.ylabel("Średnia nagroda")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    param_sets = [
        (0.5, 0.5, 0.15, 3),
        (0.5, 0.5, 0.15, 4),
        (0.5, 0.5, 0.15, 5),
        (0.3, 0.5, 0.15, 4),
        (0.8, 0.5, 0.15, 4),
        (0.8, 0.3, 0.15, 4),
        (0.8, 0.8, 0.15, 4),
        (0.8, 0.3, 0.15, 5),
        (0.8, 0.8, 0.15, 5),
    ]

    # Generate data files (run this once, then comment out if not needed)
    # for lr, df, er, b in param_sets:
    #     gen_data(10, lr, df, er, b)

    # Read data for visualization
    datasets = []
    for lr, df, er, b in param_sets:
        label = f"lr={lr}, df={df}, er={er}, b={b}"
        _, avg, std = read_data(lr, df, er, b)
        datasets.append((label, avg, std))

    plot_all(datasets)


if __name__ == '__main__':
    main()
