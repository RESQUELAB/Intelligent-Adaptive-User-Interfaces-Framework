import numpy as np
import ui_adapt
import gym
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
import datetime

class QLearningAgent:

    def __init__(self, env, QTABLE_PATH = "rl_models/qtable.pickle",
                 SIGMA=1, LEARNING_RATE=0.85, DISCOUNT=0.8, EPISODES=60000, 
                 SHOW_EVERY=2000, UPDATE_EVERY=150,
                 MAX_STEPS=20, MAX_STEPS_TARGET=3,
                 START_EPSILON_DECAYING=1, END_EPSILON_DECAYING=None, 
                 MIN_EPSILON=0.1):
        if isinstance(env, str): 
            print("IT IS A STRING :: ", env)
            self.env_name = env
            self.env = gym.make(self.env_name)
        else:
            print("IT IS AN ENV :: ", env)
            print("\t..UIDesign ", env.uidesign)
            print("\t..UIDesign ", env.uidesign.server_socket)
            self.env = env
        if QTABLE_PATH:            
            with open(QTABLE_PATH, 'rb') as file:
                self.saved_data = pickle.load(file)
        self.SIGMA = SIGMA
        self.LEARNING_RATE = LEARNING_RATE
        self.DISCOUNT = DISCOUNT
        self.EPISODES = EPISODES
        self.SHOW_EVERY = SHOW_EVERY
        self.UPDATE_EVERY = UPDATE_EVERY
        self.MAX_STEPS = MAX_STEPS
        self.MAX_STEPS_TARGET = MAX_STEPS_TARGET
        self.epsilon = 1
        self.START_EPSILON_DECAYING = START_EPSILON_DECAYING
        self.END_EPSILON_DECAYING = END_EPSILON_DECAYING if END_EPSILON_DECAYING is not None else EPISODES // 2
        self.MIN_EPSILON = MIN_EPSILON
        self.obsSpaceSize, self.qTable = self.create_q_table()
        self.metrics = {'ep': [], 'avg_step': [], 'min_step': [], 'max_step': [],
                        'avg_rew': [], 'min_rew': [], 'max_rew': [],
                        'epsilon': [], 'action': [],
                        'rolling_averages_eps': [],
                        'rolling_averages': []}
        self.epsilon_plot = []
        self.firstEpisode = True
        self.episode_scores = []
        self.rolling_averages = []
        self.rolling_averages_eps = []
        self.rolling_window_size = 150

    def create_q_table(self):
        obsSpaceSize = self.env.observation_space.nvec
        action_space_size = self.env.action_space.n
        qTable = np.zeros((np.prod(obsSpaceSize), action_space_size))
        return obsSpaceSize, qTable

    def get_discrete_state(self, state):
        state_idx = np.ravel_multi_index(state, self.obsSpaceSize)
        return state_idx

    def train(self):
        for episode in range(self.EPISODES):
            discreteState = self.get_discrete_state(self.env.reset())
            done = False
            step = 0

            while not done:
                if episode % self.SHOW_EVERY == 0:
                    pass
                
                step += 1
                if np.random.random() > self.epsilon:
                    action = np.argmax(self.qTable[discreteState])
                else:
                    action = np.random.randint(0, self.env.action_space.n)
                
                newState, reward, done, _ = self.env.step(action, sigma=self.SIGMA)
                newDiscreteState = self.get_discrete_state(newState)
                maxFutureQ = np.max(self.qTable[newDiscreteState])
                currentQ = self.qTable[discreteState][action]

                if step > self.MAX_STEPS:
                    break
                elif done and step <= self.MAX_STEPS_TARGET:
                    reward += 1

                newQ = (1 - self.LEARNING_RATE) * currentQ + self.LEARNING_RATE * (reward + self.DISCOUNT * maxFutureQ)
                self.qTable[discreteState][action] = newQ
                discreteState = newDiscreteState
            
            self.episode_scores.append(reward)

            if episode >= self.rolling_window_size - 1:
                rolling_average = pd.Series(self.episode_scores).rolling(window=self.rolling_window_size).mean().values[-1]
                self.rolling_averages.append(rolling_average)
                self.rolling_averages_eps.append(episode)

            if episode % self.UPDATE_EVERY == 0:
                if self.firstEpisode:
                    self.firstEpisode = False
                    continue
                latestEpisodes = self.episode_scores[-self.UPDATE_EVERY:]
                averageStep = sum(latestEpisodes) / len(latestEpisodes)
                averageRew = sum(latestEpisodes) / len(latestEpisodes)
                averageAction = action
                self.metrics['ep'].append(episode)
                self.metrics['action'].append(averageAction)
                self.metrics['avg_step'].append(averageStep)
                self.metrics['min_step'].append(min(latestEpisodes))
                self.metrics['max_step'].append(max(latestEpisodes))
                self.metrics['avg_rew'].append(averageRew)
                self.metrics['min_rew'].append(min(latestEpisodes))
                self.metrics['max_rew'].append(max(latestEpisodes))
                self.metrics['epsilon'].append(self.epsilon)
                print("Episode:", episode, "/", str(self.EPISODES), "\n\tSteps - Average:", averageStep, "Min:", min(latestEpisodes), "Max:", max(latestEpisodes))
                print("\tReward - Average:", averageRew, "Min:", min(latestEpisodes), "Max:", max(latestEpisodes))
                self.epsilon_plot.append(self.epsilon)

            if self.END_EPSILON_DECAYING >= episode >= self.START_EPSILON_DECAYING:
                self.epsilon -= (self.epsilon / (self.END_EPSILON_DECAYING - self.START_EPSILON_DECAYING))
                if self.epsilon <= self.MIN_EPSILON:
                    self.epsilon = self.MIN_EPSILON
        
        self.env.close()

    def save_model(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        models_dir = 'RL_algorithms/models'
        filename = f"QLearning_Sigma_{self.SIGMA}_{self.env_name}.pickle"
        file_path = os.path.join(models_dir, filename)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        with open(file_path, 'wb') as file:
            pickle.dump(self.qTable, file)

    def load_q_table(self):
        obsSpaceSize = self.env.observation_space.nvec
        qTable = self.saved_data['q_table']
        return obsSpaceSize, qTable

    def get_discrete_state(self, state):
        state_idx = np.ravel_multi_index(state, self.obsSpaceSize)
        return state_idx

    def evaluate(self):
        for episode in range(self.EPISODES):
            discreteState = self.get_discrete_state(self.env.reset())
            done = False
            step = 0

            while not done:
                step += 1
                action = np.argmax(self.q_table[discreteState])
                newState, reward, done, info = self.env.step(action, sigma=self.SIGMA)
                newDiscreteState = self.get_discrete_state(newState)
                alignment = info["alignment"]

                if step > self.MAX_STEPS:
                    break
                elif done and step <= self.MAX_STEPS_TARGET:
                    reward += 1
                discreteState = newDiscreteState

            self.previousStep.append(step)
            self.previousReward.append(round(reward, 4))
            self.previousAlignment.append(round(alignment, 4))
            self.episode_scores.append(reward)

            if episode >= self.rolling_window_size - 1:
                rolling_average = pd.Series(self.episode_scores).rolling(window=self.rolling_window_size).mean().values[-1]
                self.rolling_averages.append(rolling_average)
                self.rolling_averages_eps.append(episode)

            if episode % self.UPDATE_EVERY == 0:
                if self.firstEpisode:
                    self.firstEpisode = False
                    continue
                latestEpisodes = self.previousStep[-self.UPDATE_EVERY:]
                latestEpisodes_reward = self.previousReward[-self.UPDATE_EVERY:]
                latestEpisodes_alignment = self.previousAlignment[-self.UPDATE_EVERY:]
                averageStep = round(sum(latestEpisodes) / len(latestEpisodes), 4)
                averageRew = round(sum(latestEpisodes_reward) / len(latestEpisodes_reward), 4)
                averageAlignment = round(sum(latestEpisodes_alignment) / len(latestEpisodes_alignment), 4)
                self.metrics['ep'].append(episode)
                self.metrics['avg_step'].append(averageStep)
                self.metrics['min_step'].append(min(latestEpisodes))
                self.metrics['max_step'].append(max(latestEpisodes))
                self.metrics['avg_rew'].append(averageRew)
                self.metrics['min_rew'].append(min(latestEpisodes_reward))
                self.metrics['max_rew'].append(max(latestEpisodes_reward))
                self.metrics['avg_alig'].append(averageAlignment)

                print("Episode:", episode, "/", str(self.EPISODES), "\n\tSteps - Average:", averageStep, "Min:", min(latestEpisodes), "Max:", max(latestEpisodes))
                print("\tReward - Average:", averageRew, "Min:", min(latestEpisodes_reward), "Max:", max(latestEpisodes_reward))
                print("\tAlignment - Average:", averageAlignment, "Min:", min(latestEpisodes_alignment), "Max:", max(latestEpisodes_alignment))
        self.env.close()

    def plot_results(self):
        plt.plot(self.rolling_averages, label=f'Rolling Average (Window {self.rolling_window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Rolling Average')
        plt.legend()
        plt.show()

        fig, ax = plt.subplots(2, 2)

        ax[0][1].plot(self.metrics['ep'], self.metrics['avg_step'], label="average steps")
        ax[0][1].set_title("Testing - Average number of steps")

        ax[1][0].plot(self.metrics['ep'], self.metrics['avg_rew'], label="average rewards")
        ax[1][0].set_title("Testing - Average Reward")

        ax[1][1].plot(self.metrics['ep'], self.metrics['avg_alig'], label="average alignment")
        ax[1][1].set_title("Testing - Average Alignment per Episode")

        plt.show()

    def save_results(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        models_dir = 'RL_algorithms/evaluation'
        filename = "QLearning_EVAL_SIGMA" + str(self.SIGMA) + "_" + self.env_name + ".csv"
        file_path = os.path.join(models_dir, filename)
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        keys = self.metrics.keys()
        rows = zip(*self.metrics.values())

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(keys)
            writer.writerows(list(rows))

        print(f'The dictionary has been saved to {file_path}.')

# Usage
# file_path = 'RL_algorithms/models/QLearning_Sigma_0.5_UIAdaptation-v0.pickle'
# SIGMA = 0.5
# env_name = "UIAdaptation-v0"

# agent = QLearningAgent(file_path, SIGMA, env_name)
# agent.train()
# agent.plot_results()
# agent.save_results()
