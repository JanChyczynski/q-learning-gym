import math
from random import random
from typing import Tuple

import gym
import time

import numpy as np
import csv
import matplotlib.pyplot as plt

ITERATIONS = 4000
MOVING_AVG_WINDOW = 200


class QLearner:
    def __init__(self, learning_rate, discount_factor, experiment_rate, discretization_buckets, lr_min, er_min, lr_decay, er_decay):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.experiment_rate = experiment_rate
        self.discretization_buckets = discretization_buckets
        self.lr_min = lr_min
        self.er_min = er_min
        self.lr_decay = lr_decay
        self.er_decay = er_decay
        self.default_q = 0
        self.environment = gym.make("CartPole-v1", render_mode=None)
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

    def qdict_histograms(self, iterations, tail_reward_avg, name):
        q_values = list(self.q_dict.values())
        plt.figure(figsize=(10, 5))
        plt.hist(q_values, bins=50, alpha=0.7, color='blue')
        plt.title(f"Wartości Q: {name} | i={iterations}, tail avg={tail_reward_avg}")
        plt.xlabel("Wartość Q")
        plt.ylabel("Liczba wystąpień")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def learn(self, max_attempts):
        name = f"Params: lr={self.learning_rate:.4f}, df={self.discount_factor}, er={self.experiment_rate:.4f}, b={self.discretization_buckets}"
        print(
            f"Params: lr={self.learning_rate:.4f}, df={self.discount_factor}, er={self.experiment_rate:.4f}, b={self.discretization_buckets}: STARTING ")
        rewards = [0] * max_attempts
        cooldown_low = 0
        cooldown_high = 0
        for i in range(max_attempts):
            reward_sum = self.attempt()
            rewards[i] = reward_sum
            if i > 50:
                # if i % 50 == 0 or 1980 < i < 2030:
                #     print(f"i: {i}, avg tail: {tail_rewards_avg}, std: {tail_rewards_std}, current rew: {reward_sum}" )

                tail_rewards = rewards[max(0, i - 20):i]
                tail_rewards_avg = np.mean(tail_rewards)
                tail_rewards_std = np.std(tail_rewards)
                if tail_rewards_avg < 50 and cooldown_low == 0:
                    print(i, "LOW tail_rewards_avg < 30., tail avg:", tail_rewards_avg, " std:", tail_rewards_std)
                    # self.qdict_histograms(i, tail_rewards_avg, name)  # call qdict_histograms
                    cooldown_low = 400
                    cooldown_high = 0
                elif tail_rewards_avg > 350 and cooldown_high == 0:
                    print(i, "HIGH tail_rewards_avg > 350, tail avg:", tail_rewards_avg, " std:", tail_rewards_std)
                    # self.qdict_histograms(i, tail_rewards_avg, name)  # call qdict_histograms
                    cooldown_high = 400
                    cooldown_low = 0
                # elif i == 2025:
                #     print(i, "2025 tail_rewards_avg ???, tail avg:", tail_rewards_avg, " std:", tail_rewards_std)
                    # self.qdict_histograms(i, tail_rewards_avg, name)  # call qdict_histograms


            cooldown_low = max(0, cooldown_low -1)
            cooldown_high = max(0, cooldown_high -1)
            self.learning_rate = max(self.lr_min, self.learning_rate * self.lr_decay)
            self.experiment_rate = max(self.er_min, self.experiment_rate * self.er_decay)

        tail = rewards[-100:]
        avg_tail = np.mean(tail)
        std_tail = np.std(tail)
        print(
            f"Params: lr={self.learning_rate:.4f}, df={self.discount_factor}, er={self.experiment_rate:.4f}, b={self.discretization_buckets} => avg: {avg_tail:.2f}, std: {std_tail:.2f}")

        self.environment.reset()
        return rewards

    def attempt(self):
        observation = self.discretise(self.environment.reset()[0])
        done = False
        reward_sum = 0.
        truncated = False
        while not (done or truncated):
            action = self.pick_action(observation)
            new_observation, reward, done, truncated, info = self.environment.step(action)
            new_observation = self.discretise(new_observation)
            self.update_knowledge(action, observation, new_observation, reward)
            observation = new_observation
            reward_sum += reward
        self.attempt_no += 1
        return reward_sum

    def discretise(self, observation):
        scaled = [(val - low) / (up - low) for val, low, up in zip(observation, self.lower_bounds, self.upper_bounds)]
        clipped = [np.clip(s, 0, 1) for s in scaled]
        return [int(round(c * (self.discretization_buckets - 1))) for c in clipped]

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

def compute_decay_rate(start_value, end_value, target_iterations):
    return (end_value / start_value) ** (1 / target_iterations)

def gen_data(experiments, learning_rate, discount_factor, experiment_rate, discretization_buckets, lr_min, er_min, lr_decay, er_decay):

    results = []
    # results_backup = []
    for _ in range(experiments):
        learner = QLearner(learning_rate, discount_factor, experiment_rate, discretization_buckets, lr_min, er_min, lr_decay, er_decay)
        curr_result = learner.learn(ITERATIONS)
        results.append(curr_result)
        # results_backup.append(curr_result.copy())
    # results = np.array(results).reshape(ITERATIONS, -1)
    avg = np.average(results, axis=0)
    std = np.std(results, axis=0)

    filename = f"data_lr{learning_rate}_lrmin{lr_min}_lrdec{lr_decay}_df{discount_factor}_er{experiment_rate}_ermin{er_min}_erdec{er_decay}_b{discretization_buckets}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["avg", "std"])
        for a, s in zip(avg, std):
            writer.writerow([a, s])

def read_data(learning_rate, discount_factor, experiment_rate, discretization_buckets, lr_min, er_min, lr_decay, er_decay):
    filename = f"data_lr{learning_rate}_lrmin{lr_min}_lrdec{lr_decay}_df{discount_factor}_er{experiment_rate}_ermin{er_min}_erdec{er_decay}_b{discretization_buckets}.csv"
    avg, std = [], []
    with open(filename, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            avg.append(float(row["avg"]))
            std.append(float(row["std"]))
    return filename, np.array(avg), np.array(std)

def plot_all(datasets):
    for label, avg, std in datasets:
        plt.figure(figsize=(12, 8))
        x = np.arange(len(avg))
        ma = moving_average(avg, MOVING_AVG_WINDOW)
        std_ma = moving_average(std, MOVING_AVG_WINDOW)
        x_ma = np.arange(len(ma))

        plt.plot(x_ma, ma, label=label, linewidth=2)
        plt.plot(x_ma, ma + std_ma, linestyle='--', linewidth=1, alpha=0.5, color='gray')
        plt.plot(x_ma, ma - std_ma, linestyle='--', linewidth=1, alpha=0.5, color='gray')

        plt.title("Średnie wyniki z korytarzem odchyleń")
        plt.xlabel("Iteracja")
        plt.ylabel("Średnia nagroda")
        plt.grid(True)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout()
        plt.show()

    # Zbiorczy wykres tylko średnich
    plt.figure(figsize=(10, 10))
    for label, avg, _ in datasets:
        ma = moving_average(avg, MOVING_AVG_WINDOW)
        x_ma = np.arange(len(ma))
        plt.plot(x_ma, ma, label=label, linewidth=2)

    plt.title("Zbiorczy wykres średnich")
    plt.xlabel("Iteracja")
    plt.ylabel("Średnia nagroda")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    param_sets = [

        # (0.001, 1.0, 0.995, 5, 0.05, 0.01, 0.999, 0.999),
        # (0.05, 1.0, 0.9, 7, 0.05, 0.01, 0.999, 0.999),
        # (0.05, 1.0, 0.9, 11, 0.05, 0.01, 0.999, 0.999),
        # (0.1, 1.0, 0.9, 7, 0.1, 0.01, 0.999, 0.999),
        # (0.1, 1.0, 0.9, 11, 0.1, 0.01, 0.999, 0.999),
        # (0.05, 0.995, 0.9, 7, 0.05, 0.01, 0.999, 0.999),
        # (0.05, 0.995, 0.9, 11, 0.05, 0.01, 0.999, 0.999),
        (0.1, 0.995, 0.9, 7, 0.1, 0.01, 0.999, 0.999),
        # (0.1, 0.995, 0.9, 11, 0.1, 0.01, 0.999, 0.999),
        # (0.05, 0.99, 0.9, 7, 0.05, 0.01, 0.999, 0.999),
        # (0.05, 0.99, 0.9, 11, 0.05, 0.01, 0.999, 0.999),
        # (0.1, 0.99, 0.9, 7, 0.1, 0.01, 0.999, 0.999),
        # (0.1, 0.99, 0.9, 11, 0.1, 0.01, 0.999, 0.999),

        (0.2, 0.995, 0.9, 7, 0.2, 0.01, 0.999, 0.999),
        (0.2, 0.995, 0.9, 11, 0.2, 0.01, 0.999, 0.999),
        (0.5, 0.995, 0.9, 7, 0.001, 0.01, 0.999, 0.999),
        (0.5, 0.995, 0.9, 11, 0.001, 0.01, 0.999, 0.999),
        (0.2, 0.995, 0.9, 7, 0.001, 0.01, 0.999, 0.999),
        (0.2, 0.995, 0.9, 11, 0.001, 0.01, 0.999, 0.999),

        (0.1, 0.995, 0.9, 7, 0.001, 0.01, 0.999, 0.999),
        (0.1, 0.995, 0.9, 9, 0.001, 0.01, 0.999, 0.999),
        (0.1, 0.995, 0.9, 11, 0.001, 0.01, 0.999, 0.999),

        # (0, 1, 1, 5, 0, 1, 0.999, 0.999),
    ]

    # Generate data files (run this once, then comment out if not needed)
    print("Write anything to generate data")
    print(input())
    for lr, df, er, b, lr_min, er_min, lr_decay, er_decay in param_sets:
        gen_data(2, lr, df, er, b, lr_min, er_min, lr_decay, er_decay)

    # Read data for visualization
    datasets = []
    for lr, df, er, b, lr_min, er_min, lr_decay, er_decay in param_sets:
        label = f"lr={lr}, df={df}, er={er}, b={b}, lr_min={lr_min}, er_min={er_min}"
        _, avg, std = read_data(lr, df, er, b, lr_min, er_min, lr_decay, er_decay)
        datasets.append((label, avg, std))

    plot_all(datasets)


if __name__ == '__main__':
    main()
