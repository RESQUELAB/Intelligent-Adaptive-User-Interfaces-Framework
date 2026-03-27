import pickle
import os
import pandas as pd

class RewardPredictor:
    def __init__(self, filename, mode=None, env=None):
        self.mode = mode
        print(" self.mode ::: ", self.mode)
        if self.mode =="RLHF":
            pass
            # self.model = OrdinalRewardModel(
            #         'human', env, "testadaptivesports", 4)
            # self.model.try_to_load_model_from_checkpoint()
        elif self.mode == "HCI":
            pass
        self.number_of_decimals = 4    
        current_folder = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_folder, filename)
        print(f"model path {filename} tttt")
        self.model = pickle.load(open(model_path, 'rb'))

    def predict(self, data=None, path=None):
        if self.mode =="RLHF":
            predictions = self.model.predict_reward(path)
        elif self.mode == "HCI":
            data = self.prepare_data(data)
            predictions = self.model.predict(data)

        # Apply rounding to avoid large number of decimals
        # Apply also x1.5 scaling to fit rewards with the 0 to 1 values.
        for i, prediction in enumerate(predictions):
            predictions[i] = predictions[i] * 2 if predictions[i] * 1.5 <= 1 else 1
            predictions[i] = round(predictions[i], self.number_of_decimals)
        return predictions
    
    def prepare_data(self,data):
        # data is the pointer to the UIDesign class instantiation
        data_columns = ['theme_dark', 'theme_light', 'display_grid', 'display_list']
        data_values = []
        if data.theme == 'light':
            data_values.extend([0,1])
        elif data.theme == 'dark':
            data_values.extend([1,0])

        if hasattr(data, 'layout'):
            if 'grid' in data.layout:
                data_values.extend([1,0])
            elif data.layout == 'list':
                data_values.extend([0,1])
        elif hasattr(data, 'display'):
            if 'grid' in data.display:
                data_values.extend([1,0])
            elif data.display == 'list':
                data_values.extend([0,1])
        
        # Create a new DataFrame for making predictions
        new_data = pd.DataFrame([data_values], columns=data_columns)
        return new_data

    def get_alignment(self, user, uidesign):
        # How different are the user preferences and the uidesign.
        
        max_alignment = len(user.preferences)

        alignment = max_alignment  # Start with the maximum alignment

        for attribute, attribute_value in uidesign.attributes.items():
            user_value = user.preferences.get(attribute, "").lower()
            attribute_value_lower = attribute_value.lower()
            if user_value != attribute_value_lower:
                alignment -= 1
         # Normalize alignment score to be between 0 and 1
        normalized_alignment = alignment / max_alignment
        return normalized_alignment
    

import os
import random
from time import sleep

from multiprocessing import Process

import numpy as np
import tensorflow as tf
from keras import backend as K
from scipy import stats

from ui_adapt.envs.nn import FullyConnectedMLP, SimpleConvolveObservationQNet

def nn_predict_rewards(obs_segments, act_segments, network, obs_shape, act_shape):
    """
    :param obs_segments: tensor with shape = (batch_size, segment_length) + obs_shape
    :param act_segments: tensor with shape = (batch_size, segment_length) + act_shape
    :param network: neural net with .run() that maps obs and act tensors into a (scalar) value tensor
    :param obs_shape: a tuple representing the shape of the observation space
    :param act_shape: a tuple representing the shape of the action space
    :return: tensor with shape = (batch_size, segment_length)
    """
    batchsize = tf.shape(obs_segments)[0]
    segment_length = tf.shape(obs_segments)[1]

    # Temporarily chop up segments into individual observations and actions
    obs = tf.reshape(obs_segments, (-1,) + obs_shape)
    acts = tf.reshape(act_segments, (-1,) + act_shape)

    # Run them through our neural network
    rewards = network.run(obs, acts)

    # Group the rewards back into their segments
    return tf.reshape(rewards, (batchsize, segment_length))

class RewardModel(object):
    def __init__(self, episode_logger=None):
        self._episode_logger = episode_logger

    def predict_reward(self, path):
        raise NotImplementedError()  # Must be overridden

    def path_callback(self, path):
        self._episode_logger.log_episode(path)

    def train(self, iterations=1, report_frequency=None):
        pass  # Doesn't require training by default

    def save_model_checkpoint(self):
        pass  # Nothing to save

    def try_to_load_model_from_checkpoint(self):
        pass  # Nothing to load

class OriginalEnvironmentReward(RewardModel):
    """Model that always gives the reward provided by the environment."""

    def predict_reward(self, path):
        return path["original_rewards"]

class OrdinalRewardModel(RewardModel):
    """A learned model of an environmental reward using training data that is merely sorted."""

    def __init__(self, model_type, env, experiment_name, stacked_frames):
        # TODO It's pretty asinine to pass in env, env_id, and make_env. Cleanup!
        super().__init__()

        self.experiment_name = experiment_name

        # Build and initialize our model
        config = tf.ConfigProto(
            device_count={'GPU': 0},
            log_device_placement=True,
        )
        config.gpu_options.per_process_gpu_memory_fraction = 0.35  # allow_growth = True
        self.sess = tf.Session(config=config)

        self.obs_shape = env.observation_space.shape
        if stacked_frames > 0:
            self.obs_shape = self.obs_shape + (stacked_frames,)
        self.discrete_action_space = (len(env.action_space.shape) == 0)
        self.act_shape = (env.action_space.n,) if self.discrete_action_space else env.action_space.shape

        self.graph = self._build_model()
        self.sess.run(tf.global_variables_initializer())
        my_vars = tf.global_variables()
        self.saver = tf.train.Saver({var.name: var for var in my_vars}, max_to_keep=0)

    def _build_model(self):
        """Our model takes in path segments with observations and actions, and generates rewards (Q-values)."""
        # Set up observation placeholder
        self.obs_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, None) + self.obs_shape, name="obs_placeholder")

        # Set up action placeholder
        if self.discrete_action_space:
            self.act_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, None), name="act_placeholder")
            # Discrete actions need to become one-hot vectors for the model
            segment_act = tf.one_hot(tf.cast(self.act_placeholder, tf.int32), self.act_shape[0])
            # HACK Use a convolutional network for Atari
            # TODO Should check the input space dimensions, not the output space!

            if len(self.obs_shape) == 3:
                net = SimpleConvolveObservationQNet(self.obs_shape, self.act_shape)
            else:
                # Use original obs_shape for 1D vector
                net = FullyConnectedMLP(self.obs_shape, self.act_shape)
        else:
            self.act_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, None) + self.act_shape, name="act_placeholder")
            # Assume the actions are how we want them
            segment_act = self.act_placeholder
            # In simple environments, default to a basic Multi-layer Perceptron (see TODO above)
            net = FullyConnectedMLP(self.obs_shape, self.act_shape)

        # Our neural network maps a (state, action) pair to a reward
        self.rewards = nn_predict_rewards(self.obs_placeholder, segment_act, net, self.obs_shape, self.act_shape)

        # We use trajectory segments rather than individual (state, action) pairs because
        # video clips of segments are easier for humans to evaluate
        self.segment_rewards = tf.reduce_sum(self.rewards, axis=1)

        self.targets = tf.placeholder(dtype=tf.float32, shape=(None,), name="reward_targets")

        self.loss = tf.reduce_mean(tf.square(self.targets - self.segment_rewards))

        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        return tf.get_default_graph()

    def predict_reward(self, path):
        """Predict the reward for each step in a given path"""
        with self.graph.as_default():
            predicted_rewards = self.sess.run(self.rewards, feed_dict={
                self.obs_placeholder: np.asarray([path["obs"]]),
                self.act_placeholder: np.asarray([path["actions"]]),
                K.learning_phase(): False
            })
        return predicted_rewards[0]  # The zero here is to get the single returned path.

    def _checkpoint_filename(self):
        return 'checkpoints\\reward_model\\%s\\treesave' % (self.experiment_name)

    def save_model_checkpoint(self):
        print("Saving reward model checkpoint!")
        self.saver.save(self.sess, self._checkpoint_filename())

    def try_to_load_model_from_checkpoint(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename()))
        if filename is None:
            print('No reward model checkpoint found on disk for experiment "{}"'.format(self.experiment_name))
        else:
            self.saver.restore(self.sess, filename)
            print("Reward model loaded from checkpoint!")
            