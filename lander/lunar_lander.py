import math
from random import random

import gym
import os
from datetime import datetime

import numpy as np
import csv
import matplotlib.pyplot as plt

ITERATIONS = 15000
MOVING_AVG_WINDOW = 800


class QLearner:
    def __init__(self, learning_rate, discount_factor, experiment_rate, discretization_buckets, lr_min, er_min, lr_decay, er_decay, sarsa):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.experiment_rate = experiment_rate
        self.discretization_buckets = discretization_buckets
        self.lr_min = lr_min
        self.er_min = er_min
        self.lr_decay = lr_decay
        self.er_decay = er_decay
        self.default_q = 0
        self.environment = gym.make("LunarLander-v2", render_mode=None)
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
        self.sarsa = sarsa
        self.q_dict = {}
        self.prev_action = None
        self.prev_observation = None

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
        algorithm_type = "SARSA" if self.sarsa else "Classic"
        name = f"{algorithm_type} Params: lr={self.learning_rate:.4f}, df={self.discount_factor}, er={self.experiment_rate:.4f}, b={self.discretization_buckets}"
        print(name + ": STARTING")
        rewards = [0] * max_attempts
        cooldown_low = 0
        cooldown_high = 0
        for i in range(max_attempts):
            reward_sum = self.attempt()
            rewards[i] = reward_sum
            if False and i > 50:
                tail_rewards = rewards[max(0, i - 20):i]
                tail_rewards_avg = np.mean(tail_rewards)
                tail_rewards_std = np.std(tail_rewards)
                if tail_rewards_avg < 50 and cooldown_low == 0:
                    print(i, "LOW tail_rewards_avg < 30., tail avg:", tail_rewards_avg, " std:", tail_rewards_std)
                    self.qdict_histograms(i, tail_rewards_avg, name)
                    cooldown_low = 400
                    cooldown_high = 0
                elif tail_rewards_avg > 350 and cooldown_high == 0:
                    print(i, "HIGH tail_rewards_avg > 350, tail avg:", tail_rewards_avg, " std:", tail_rewards_std)
                    self.qdict_histograms(i, tail_rewards_avg, name)
                    cooldown_high = 400
                    cooldown_low = 0

            cooldown_low = max(0, cooldown_low -1)
            cooldown_high = max(0, cooldown_high -1)
            self.learning_rate = max(self.lr_min, self.learning_rate * self.lr_decay)
            self.experiment_rate = max(self.er_min, self.experiment_rate * self.er_decay)

        tail = rewards[-100:]
        avg_tail = np.mean(tail)
        std_tail = np.std(tail)
        print(name + f" => avg: {avg_tail:.2f}, std: {std_tail:.2f}")

        self.environment.reset()
        return rewards

    def attempt(self):
        prev_reward = None
        new_action = None
        observation = self.discretise(self.environment.reset()[0])
        done = False
        reward_sum = 0.
        truncated = False
        while not (done or truncated):
            action = new_action if new_action is not None else self.pick_action(observation)
            new_observation, reward, done, truncated, info = self.environment.step(action)
            new_observation = self.discretise(new_observation)
            new_action = self.pick_action(new_observation)
            if not self.sarsa:
                self.update_knowledge(action, observation, new_observation, reward)
            elif self.prev_action is not None and self.prev_observation is not None:
                self.SARSA_update_knowledge(action, new_action, observation, new_observation, reward)
                # self.SARSA_update_knowledge(self.prev_action, action, self.prev_observation, new_observation, reward)
            observation = new_observation
            reward_sum += reward
            self.prev_action = action
            self.prev_observation = new_observation
            prev_reward = reward
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

    def SARSA_update_knowledge(self, prev_action, action, prev_observation, observation, reward):
        old_past_q = self.q_dict.get((tuple(prev_observation), prev_action), self.default_q)
        current_q = self.q_dict.get((tuple(observation), action), self.default_q)
        new_past_q = old_past_q + self.learning_rate * (reward + self.discount_factor * current_q - old_past_q)
        self.q_dict[(tuple(prev_observation), prev_action)] = new_past_q

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

def gen_data(experiments, learning_rate, discount_factor, experiment_rate, discretization_buckets, lr_min, er_min, lr_decay, er_decay, sarsa):
    results = []
    best_learner = None
    best_avg_tail = -np.inf
    for _ in range(experiments):
        learner = QLearner(learning_rate, discount_factor, experiment_rate, discretization_buckets, lr_min, er_min, lr_decay, er_decay, sarsa=sarsa)
        curr_result = learner.learn(ITERATIONS)
        results.append(curr_result)
        avg_tail = np.mean(curr_result[-100:])
        if avg_tail > best_avg_tail:
            best_avg_tail = avg_tail
            best_learner = learner

    avg = np.average(results, axis=0)
    std = np.std(results, axis=0)

    prefix = "SARSA_" if sarsa else ""
    filename = f"{prefix}data_lr{learning_rate}_lrmin{lr_min}_lrdec{lr_decay}_df{discount_factor}_er{experiment_rate}_ermin{er_min}_erdec{er_decay}_b{discretization_buckets}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["avg", "std"])
        for a, s in zip(avg, std):
            writer.writerow([a, s])

    qdict_filename = f"qDict_{prefix}lr{learning_rate}_lrmin{lr_min}_lrdec{lr_decay}_df{discount_factor}_er{experiment_rate}_ermin{er_min}_erdec{er_decay}_b{discretization_buckets}.txt"
    with open(qdict_filename, mode='w') as f:
        f.write("{\n")
        for key, value in best_learner.q_dict.items():
            f.write(f"  {key}: {value},\n")
        f.write("}\n")

def read_data(learning_rate, discount_factor, experiment_rate, discretization_buckets, lr_min, er_min, lr_decay, er_decay, sarsa):
    prefix = "SARSA_" if sarsa else ""
    filename = f"{prefix}data_lr{learning_rate}_lrmin{lr_min}_lrdec{lr_decay}_df{discount_factor}_er{experiment_rate}_ermin{er_min}_erdec{er_decay}_b{discretization_buckets}.csv"
    avg, std = [], []
    with open(filename, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            avg.append(float(row["avg"]))
            std.append(float(row["std"]))
    return filename, np.array(avg), np.array(std)

def plot_all(datasets, sarsa):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix = "SARSA" if sarsa else "Classic"
    save_dir = f"./plots/{prefix}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    for idx, (label, avg, std) in enumerate(datasets):
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

        save_path = os.path.join(save_dir, f"{prefix}_plot_{idx}.png")
        plt.savefig(save_path)
        plt.show()
        plt.close()

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

    summary_path = os.path.join(save_dir, f"{prefix}_summary.png")
    plt.savefig(summary_path)
    plt.show()
    plt.close()

def main():
    # # (0.2, 0.9995, 0.9, 15, 0.00031, 0.01, 0.9995, 0.999, False),
    # # # (0.2, 0.9995, 0.9, 17, 0.00031, 0.01, 0.9995, 0.999, False),
    # # (0.5, 0.9995, 0.9, 17, 0.0003, 0.01, 0.999, 0.999, True),
    # # #     (0.5, 0.9995, 0.9, 17, 0.00005, 0.01, 0.999, 0.999, True),
    # # #     (0.5, 0.9995, 0.9, 17, 0.00031, 0.01, 0.9995, 0.999, True),
    #
    # # (0.2, 0.9995, 0.9, 15, 0.00031, 0.01, 0.9995, 0.999, False),
    # # (0.5, 0.9995, 0.9, 17, 0.0003, 0.01, 0.999, 0.999, True),
    # (0.2, 0.9995, 0.9, 15, 0.00031, 0.0102, 0.9995, 0.995, False),
    # (0.5, 0.9995, 0.9, 17, 0.0003, 0.0102, 0.999, 0.995, True),
    # (0.2, 0.9995, 0.9, 15, 0.000051, 0.01, 0.9995, 0.999, False),
    # (0.5, 0.9995, 0.9, 17, 0.00005, 0.01, 0.999, 0.999, True),
    # (0.2, 0.9995, 0.9, 15, 0.00031, 0.01, 0.9995, 0.999, False),
    # (0.5, 0.9995, 0.9, 17, 0.0003, 0.01, 0.999, 0.999, True),
    # (0.2, 0.9995, 0.9, 15, 0.000051, 0.00301, 0.9995, 0.999, False),
    # (0.5, 0.9995, 0.9, 17, 0.00005, 0.00301, 0.999, 0.999, True),

    param_sets = [
        (0.2, 0.9995, 0.9, 15, 0.00031, 0.0102, 0.9995, 0.995, False),
        (0.5, 0.9995, 0.9, 17, 0.0003, 0.0102, 0.999, 0.995, True),
        (0.2, 0.9995, 0.9, 15, 0.00031, 0.01021, 0.9995, 0.99, False),
        (0.5, 0.9995, 0.9, 17, 0.0003, 0.01021, 0.999, 0.99, True),
        (0.2, 0.9995, 0.9, 15, 0.00031, 0.01022, 0.9995, 0.95, False),
        (0.5, 0.9995, 0.9, 17, 0.0003, 0.01022, 0.999, 0.95, True),
        (0.2, 0.9995, 0.9, 15, 0.00031, 0.0502, 0.9995, 0.995, False),
        (0.5, 0.9995, 0.9, 17, 0.0003, 0.0502, 0.999, 0.995, True),
        (0.2, 0.9995, 0.5, 15, 0.00031, 0.0102, 0.9995, 0.995, False),
        (0.5, 0.9995, 0.5, 17, 0.0003, 0.0102, 0.999, 0.995, True),
    ]


    # print("Write anything to generate data")
    # print(input())
    # for lr, df, er, b, lr_min, er_min, lr_decay, er_decay, sarsa in param_sets:
    #     gen_data(4, lr, df, er, b, lr_min, er_min, lr_decay, er_decay, sarsa)

    datasets = []
    for lr, df, er, b, lr_min, er_min, lr_decay, er_decay, sarsa in param_sets:
        label = f"{'SARSA' if sarsa else 'Classic'} lr={lr}, df={df}, er={er}, b={b}, lr_min={lr_min}, er_min={er_min}"
        _, avg, std = read_data(lr, df, er, b, lr_min, er_min, lr_decay, er_decay, sarsa)
        datasets.append((label, avg, std))

    plot_all(datasets, sarsa=False)  # Only used for naming output directory

if __name__ == '__main__':
    main()
