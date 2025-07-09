import gym
from gym import spaces
import numpy as np

import ui_adapt.utils as utils
from ui_adapt.utils import Config, StopImage
from ui_adapt.envs.reward_predictor import RewardPredictor
import time

class UIAdaptationEnv(gym.Env):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    _max_episode_steps = 20

    def __init__(self, render_mode=None, 
                 config_data='config.json',
                 initialState = None,
                 maxDepth = 4,
                 ws_client= None, 
                 ws_server= None,
                 learning = False):
        self.name = 'uiadaptation'

        self.config = Config(config_data=config_data)
        
        self.initialState = initialState

        self.user = utils.get_random_user(self.config, userinfo=initialState)
        self.uidesign = utils.get_random_ui(self.config, uiDesignInfo=initialState, client_ws=ws_client, server_ws=ws_server)
        self.platform = utils.get_random_platform(self.config, platforminfo=initialState)
        self.environment = utils.get_random_environment(self.config, envinfo=initialState)

        self.actions = utils.get_actions(self.config)

        self.action_space = spaces.Discrete(len(self.actions))

        self.all_combinations = (self.uidesign.combinations + 
                            self.user.combinations +
                           self.platform.combinations + 
                           self.environment.combinations)
        self.observation_space = gym.spaces.MultiDiscrete(self.all_combinations)
        # self.observation_space_size = np.prod(self.observation_space.nvec)
        self.observation_space_size = tuple(self.observation_space.nvec)
        
        self.state = self.get_observation()
        
        self.reward_collected = 0

        self.depth = 0
        self.maxDepth = maxDepth

        self.learning = learning

        model_path = "model.pckl"
        mode="HCI"
        if "REWARD" in self.config.config:
            if self.config.config["REWARD"]["MODE"] == "HCI":
                # CHANGE HERE THE PATH MODEL TO USE THE HCI MODEL
                model_path = "model.pckl"
                mode = "HCI"
            elif self.config.config["REWARD"]["MODE"] == "HCIHF":
                # CHANGE HERE THE PATH MODEL TO USE THE HCI+HF MODEL
                model_path = "model.pckl"
                mode = "HCI"
                # mode = "RLHF"
        self.reward_predictor = RewardPredictor(model_path, mode=mode, env=self)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def is_terminal(self):
        if self.depth >= self.maxDepth:
             return True
        return False
    
    def updateSockets(self, clientWS, serverWS):
        self.uidesign.updateSockets(clientWS, serverWS)

    def setImage(self, image):
        self.uidesign.setImage(image)

    def render(self, render_mode='human'):
        '''
        Returns: None
        Show the current environment state e.g., the graphical window
        This method must be implemented, but it's ok to have an empty implementation if
        rendering is not important
        '''
        if render_mode is not "human":
            self.uidesign.render()
            self.user.info()
            self.environment.info()
            self.platform.info()

    def close (self):
        '''
        Return: None
        This method is optional. Used to cleanup all resources (threads, GUI, etc)
        '''
        # print("CLOSING")
        pass


    def step(self, action, verbose=False, sigma=1):
        err_msg = f"Action {action!r} ({type(action)}) invalid. Does not exist in ACTION_SPACE."
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."        
        initial_state = self.get_observation()
        done = False
        info = {}
        reward = 0
        self.depth += 1
        
        penalize_flag = False

        action_data = self.actions.get(action, {})
        if not action_data:
            print(f"Action {action} is not defined.")
            return

        name = action_data.get("name", "")
        target = action_data.get("target", "")
        value = action_data.get("value", "")
        api_call = action_data.get("api_call", "")
        # print("\n\nTHIS IS THE SERVER_SOCKET: ", self.uidesign.server_socket ,"\n")
        self.uidesign.update(target, value, api_call=api_call)
        
        self.state = self.get_observation()
        if sigma != 1:
            alignment = self.reward_predictor.get_alignment(
                        self.user, self.uidesign)
        else:
            alignment = 0
        info["alignment"] = alignment
        info["stop_flag"] = False
        if self.learning:
            time.sleep(0.6)
            counter = 0
            while self.uidesign.getImage() is None:
                if counter % 10 == 0:
                    print("waiting for image.")
                    counter = 0
                counter = counter + 1
                time.sleep(0.1)
            getPrevImage = False
            if isinstance(self.uidesign.getImage(), StopImage):
                info["stop_flag"] = True
                getPrevImage = True
            
            info["human_obs"] = self.uidesign.getImage(clear=True, getPrevious=getPrevImage)


        '''if penalize_flag:
            reward = -5
        else:
            reward = self.compute_reward()         
            # reward = alignment
        '''
        path = {
            'obs': [self.state],
            'actions': [action]
        }
        reward = self.compute_reward(sigma=sigma, path=path)
        
        self.reward_collected += reward

        
        done = bool(reward>=0.66)

        if verbose:
            print(f"Performed action: {name}, Target: {target}, Value: {value}, Reward: {reward}," 
                  f"Collected Reward: {self.reward_collected}, Done: {done}")

        return self.state, reward, done, info


    def reset(self, *, seed = None, options = None):
        '''
        We first clean everithing and then we create a new Context, app, etc
        '''
        # self.close()
        server_socket = self.uidesign.server_socket if hasattr(self.uidesign, 'server_socket') else None
        client_socket = self.uidesign.client_socket if hasattr(self.uidesign, 'client_socket') else None
        self.user = utils.get_random_user(self.config)
        self.platform = utils.get_random_platform(self.config)
        self.environment = utils.get_random_environment(self.config)
        self.uidesign = utils.get_random_ui(self.config, 
                                            uiDesignInfo=self.initialState, 
                                            client_ws=client_socket, 
                                            server_ws=server_socket)
        self.state = self.get_observation()
        self.reward_collected = 0
        self.depth = 0
        return self.state

    def state_as_array(self, state, npArray=False):
        state_array = []
        for a in state:
            if type(state[a]) is dict:
                for b in state[a]:
                    if type(state[a][b]) is dict:
                        for c in state[a][b]:
                            state_array.append(state[a][b][c])
                    else:
                        state_array.append(state[a][b])
            else:
                state_array.append(state[a])
        if npArray:
            return np.array(state_array)
        return state_array
    
    def get_observation(self):
        """
            This method traduces the representation of the state into an observation
            that the gym can work with.
        """
        uidesign_state = self.uidesign.get_state()
        user_state = self.user.get_state()
        environment_state = self.environment.get_state()
        platform_state = self.platform.get_state()

        self.state = {
            **uidesign_state,
            **user_state, 
            **platform_state,
            **environment_state
            }
        # stated_idx = self.get_discrete_state(self.state_as_array(self.state, npArray=True))
        # return stated_idx
        return self.state_as_array(self.state, npArray=True)

    
    def get_discrete_state(self, state):
        # Ensure that state is a sequence (list/tuple)
        # Check if state is numpy array, convert to tuple
        if isinstance(state, np.ndarray):
            state = tuple(state)
        if not isinstance(state, (list, tuple)):
            raise ValueError("State should be a list or tuple of discrete values.")
        
        # Using np.ravel_multi_index to compute the flattened index
        state_idx = np.ravel_multi_index(state, self.observation_space_size)
        return state_idx


    def compute_reward(self, sigma=1, path=None):
        '''
        Add here the reward function.
        '''
        if sigma==1:
            return self.general_reward(path=path)
        elif sigma == 0:
            return self.individual_reward()
        else:
            return (1 - sigma) * self.individual_reward() + sigma * self.general_reward(path=path)
    
    def general_reward(self, path=None):
        return self.reward_predictor.predict(self.uidesign, path=path)[0]
    
    def individual_reward(self):
        return self.reward_predictor.get_alignment(self.user, self.uidesign)
    
