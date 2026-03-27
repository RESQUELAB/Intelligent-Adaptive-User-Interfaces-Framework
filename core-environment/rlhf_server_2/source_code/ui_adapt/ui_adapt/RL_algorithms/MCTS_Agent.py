import gym
import numpy as np
from copy import deepcopy
import random
from math import sqrt, log
import matplotlib.pyplot as plt
import ui_adapt
from ui_adapt.envs.reward_predictor import RewardPredictor, OrdinalRewardModel


class MCTS:
    def __init__(self, game_name, episodes, c=1, mode=None):
        self.GAME_NAME = game_name
        self.episodes = episodes
        self.rewards = []
        self.moving_average = []
        self.c = c
        self.mode=mode
    
    def adapt(self, config=None,initial_state=None, env=None):
        self.episodes = 1
        action = self.play_episodes(config=config,initial_state=initial_state)
        print("Took this action:: ", action, " - ", config["ACTIONS"][str(action)])
        env.step(action)
        return

    def play_episodes(self, initial_state=None, config=None):
        for e in range(self.episodes):
            reward_e = 0
            if "uiadapt" in self.GAME_NAME.lower():
                from ui_adapt.envs.uiadaptationenv import UIAdaptationEnv
                print("its UI adapt. creating based on the config.", config)
                game = UIAdaptationEnv(config_data=config, 
                                       initialState=initial_state,
                                       ws_client=None,
                                       ws_server=None)
                game.uidesign.mode = "HEADLESS"
                self.model = OrdinalRewardModel('human', game, 
                                                "testadaptivesports", 
                                                4)
                self.model.try_to_load_model_from_checkpoint()

            else:
                game = gym.make(self.GAME_NAME)


            observation = game.state
            done = False
            
            mytree = Node(game, False, 0, observation, 0, c=self.c, agent=self)

            print('episode #' + str(e+1))
            
            while not game.is_terminal():
                print("Creating the tree")
                mytree, action = self.policy_player_mcts(mytree)
                print("Tree created")
                
                observation, reward, done, _ = game.step(action)

                print("\tREWARD: ", reward)
                print("\tREWARD_E: ", reward_e)
                reward_e = reward_e + reward
                print("\tReward: ", reward)
                
                game.render() # uncomment this if you want to see your agent in action!
                        
                if game.is_terminal():
                    print('reward_e ' + str(reward_e))
                    game.close()
                    break
                
            self.rewards.append(reward_e)
            self.moving_average.append(np.mean(self.rewards[-100:]))
        return action
        # plt.plot(self.rewards, label="Reward")
        # plt.plot(self.moving_average, label="Moving Average")
        # plt.legend()
        # plt.show()
        # print('moving average: ' + str(np.mean(self.rewards[-20:])))

    def policy_player_mcts(self, mytree):
        '''
        The strategy for using the MCTS is quite simple:
            - in order to pick the best move from the current node:
            - explore the tree starting from that node for a certain number of iterations to collect reliable statistics
            - pick the node that, according to MCTS, is the best possible next action
        '''
        for i in range(MCTS_POLICY_EXPLORE):
            mytree.explore()
            
        next_tree, next_action = mytree.next()
            
        next_tree.detach_parent()
        
        return next_tree, next_action

class Node:
    def __init__(self, game, done, parent, observation, action_index, depth=0, maxD=4, c=1, agent=None):
        self.child = None
        self.T = 0
        self.N = 0
        self.D = depth
        self.maxD = maxD
        self.game = game
        self.observation = observation
        self.done = done
        self.parent = parent
        self.action_index = action_index
        self.c = c
        self.agent = agent

    def get_ucb_score(self):
        if self.N == 0:
            return float('inf')
        
        top_node = self
        if top_node.parent:
            top_node = top_node.parent
            
        return (self.T / self.N) + self.c * sqrt(log(top_node.N) / self.N) 

    def create_child(self):
        if self.game.is_terminal():
            return
    
        actions = []
        games = []
        for i in range(GAME_ACTIONS): 
            actions.append(i)
            new_game = deepcopy(self.game)
            games.append(new_game)
            
        child = {} 
        for action, game in zip(actions, games):
            observation, reward, done, _ = game.step(action)
            depth = self.D + 1
            child[action] = Node(game, done, self, observation, action, depth=depth, c=self.c, agent=self.agent)
        
        self.child = child

    def explore(self):
        current = self
        while current.child:
            child = current.child
            max_U = max(c.get_ucb_score() for c in child.values())
            actions = [a for a, c in child.items() if c.get_ucb_score() == max_U]
            if len(actions) == 0:
                print("error zero length ", max_U)
            action = random.choice(actions)
            current = child[action]
            
        if current.N < 1:
            current.T = current.T + current.rollout()
        else:
            current.create_child()
            if current.child:
                current = random.choice(current.child)
            current.T = current.T + current.rollout()
            
        current.N += 1

        parent = current
        
        while parent.parent:
            parent = parent.parent
            parent.N += 1
            parent.T = parent.T + current.T

    def rollout(self):
        if self.game.is_terminal():
            return 0
        
        v = 0
        done = False
        new_game = deepcopy(self.game)
        while not new_game.is_terminal():
            action = new_game.action_space.sample()
            prev_obs = new_game.state
            observation, reward, done, _ = new_game.step(action)
            if self.agent.mode == "HCIHF":

                # Repeat the observation four times along a new axis
                obs_data_array = np.array([prev_obs, prev_obs, observation, observation])                
                obs_data_array = np.expand_dims(obs_data_array, axis=-1)
                obs = np.transpose(obs_data_array)

                predictions = self.agent.model.predict_reward({
                    'obs': obs,
                    'actions': [action]
                })
                reward = predictions[0]
            v = v + reward
            if new_game.is_terminal():
                new_game.reset()
                new_game.close()
                break
        return v

    def next(self):
        if self.game.is_terminal():
            raise ValueError("game has ended")

        if not self.child:
            raise ValueError('no children found and game hasn\'t ended')
        
        child = self.child
        
        max_N = max(node.N for node in child.values())
        max_children = [c for a, c in child.items() if c.N == max_N]
        
        if len(max_children) == 0:
            print("error zero length ", max_N) 
            
        max_child = random.choice(max_children)
        
        return max_child, max_child.action_index

    def detach_parent(self):
        del self.parent
        self.parent = None

MCTS_POLICY_EXPLORE = 100


GAME_NAME = 'UIAdaptation-v0'
env = gym.make(GAME_NAME)
GAME_ACTIONS = env.action_space.n
if isinstance(env.observation_space, gym.spaces.MultiDiscrete):
    GAME_OBS = np.prod(env.observation_space.nvec)
else:
    GAME_OBS = env.observation_space.shape[0]

print('In the ' + GAME_NAME + ' environment there are: ' + str(GAME_ACTIONS) + ' possible actions.')
print('In the ' + GAME_NAME + ' environment the observation is composed of: ' + str(GAME_OBS) + ' values.')

env.reset()
env.close()

