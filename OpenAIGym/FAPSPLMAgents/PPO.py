# # FAPS PLMAgents

import logging
import os
import random
from collections import deque

import keras.backend as k
import numpy as np
import tensorflow as tf
from keras import Input
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam

from OpenAIGym.exception import FAPSPLMEnvironmentException

logger = logging.getLogger("FAPSPLMAgents")


class FAPSTrainerException(FAPSPLMEnvironmentException):
    """
    Related to errors with the Trainer.
    """
    pass


class PPO(object):
    """This class is the abstract class for the faps trainers"""

    def __init__(self, envs, brain_name, trainer_parameters, training, seed):
        """
        Responsible for collecting experiences and training a neural network model.

        :param envs: The FAPSPLMEnvironment.
        :param brain_name: The brain to train.
        :param trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        :param seed: Random seed.
        """
        self.brain_name = brain_name
        self.trainer_parameters = trainer_parameters
        self.is_training = training
        self.seed = seed
        self.steps = 0
        self.last_reward = 0
        self.initialized = False

        # initialize specific PPO parameters
        self.env_brains = envs
        self.state_size = 0
        self.action_size = 0
        for env_name, env in self.env_brains.items():
            self.action_size = env.action_space.n
            self.state_size = env.observation_space.n

        # self.action_space_type = envs.actionSpaceType
        self.num_layers = self.trainer_parameters['num_layers']
        self.batch_size = self.trainer_parameters['batch_size']
        self.hidden_units = self.trainer_parameters['hidden_units']
        self.replay_memory = deque(maxlen=self.trainer_parameters['memory_size'])
        self.gamma = self.trainer_parameters['gamma']  # discount rate
        self.epsilon = self.trainer_parameters['epsilon']  # exploration rate
        self.epsilon_min = self.trainer_parameters['epsilon_min']
        self.epsilon_decay = self.trainer_parameters['epsilon_decay']
        self.learning_rate = self.trainer_parameters['learning_rate']
        self.loss_clipping = self.trainer_parameters['loss_clipping']
        self.entropy_loss = self.trainer_parameters['entropy_loss']
        self.exploration_noise = self.trainer_parameters['exploration_noise']
        self.summary = self.trainer_parameters['summary_path']
        self.tensorBoard = TensorBoard(self.summary)
        self.actor_model = None
        self.critic_model = None
        self.last_prediction = None

        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediction = np.zeros((1, self.action_size))

    def __str__(self):
        return '''Proximal Policy Optimization Trainer'''

    @property
    def parameters(self):
        """
        Returns the trainer parameters of the trainer.
        """
        return self.trainer_parameters

    @property
    def get_max_steps(self):
        """
        Returns the maximum number of steps. Is used to know when the trainer should be stopped.
        :return: The maximum number of steps of the trainer
        """
        return self.trainer_parameters['max_steps'] * self.trainer_parameters['num_epoch']

    @property
    def get_step(self):
        """
        Returns the number of steps the trainer has performed
        :return: the step count of the trainer
        """
        return self.steps

    @property
    def get_last_reward(self):
        """
        Returns the last reward the trainer has had
        :return: the new last reward
        """
        return self.last_reward

    @staticmethod
    def _exponential_average(old, new, b1):
        return old * b1 + (1 - b1) * new

    def proximal_policy_optimization_loss(self, advantage, old_prediction):
        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob / (old_prob + 1e-10)
            return -k.mean(k.minimum(r * advantage, k.clip(r, min_value=1 - self.loss_clipping,
                                                           max_value=1 + self.loss_clipping) * advantage)
                           + self.entropy_loss * -(prob * k.log(prob + 1e-10)))
        return loss

    def proximal_policy_optimization_loss_continuous(self, advantage, old_prediction):
        def loss(y_true, y_pred):
            var = k.square(self.exploration_noise)
            pi = 3.1415926
            denom = k.sqrt(2 * pi * var)
            prob_num = k.exp(- k.square(y_true - y_pred) / (2 * var))
            old_prob_num = k.exp(- k.square(y_true - old_prediction) / (2 * var))

            prob = prob_num / denom
            old_prob = old_prob_num / denom
            r = prob / (old_prob + 1e-10)

            return -k.mean(k.minimum(r * advantage, k.clip(r, min_value=1 - self.loss_clipping,
                                                           max_value=1 + self.loss_clipping) * advantage))

        return loss

    def is_initialized(self):
        """
        check if the trainer is initialized
        """
        return self.initialized
    
    def _create_actor_model(self):
        a = Input(shape=(self.state_size,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(self.action_size,))

        h = Dense(self.hidden_units, activation='relu')(a)
        # h = Dropout(0.2)(h)
        for x in range(1, self.num_layers):
            h = Dense(self.hidden_units, activation='relu')(h)
            # h = Dropout(0.2)(h)
        o = Dense(self.action_size, activation='softmax', name="output_actor_network")(h)
        model = Model(inputs=[a, advantage, old_prediction], outputs=o)
        return model, advantage, old_prediction

    def _create_critic_model(self):
        s = Input(shape=(self.state_size,))
        h = Dense(self.hidden_units, activation='relu')(s)
        # h1 = Dropout(0.2)(h1)

        for x in range(1, self.num_layers):
            h = Dense(self.hidden_units, activation='relu')(h)
            # h = Dropout(0.2)(h)

        v = Dense(1,  name="critic_output_layer")(h)
        model = Model(inputs=s, outputs=v)

        return model, s

    def initialize(self):
        """
        Initialize the trainer
        """
        self.actor_model, advantage, old_prediction = self._create_actor_model()
        self.actor_model.compile(
            loss=self.proximal_policy_optimization_loss(advantage=advantage, old_prediction=old_prediction),
            metrics=['mse'], optimizer=Adam(lr=self.learning_rate)
        )
        print('\n##### Actor Model ')
        print(self.actor_model.summary())

        self.critic_model, _ = self._create_critic_model()
        self.critic_model.compile(loss='mse', metrics=['mse'], optimizer=Adam(lr=self.learning_rate))
        print('\n##### Critic Model ')
        print(self.critic_model.summary())

        self.tensorBoard.set_model(self.actor_model)

        self.initialized = True

    def clear(self):
        """
        Clear the trainer
        """
        k.clear_session()
        self.replay_memory.clear()
        self.actor_model = None
        self.critic_model = None

    def load_model_and_restore(self, model_path):
        """
        Load and restore the model from a defined path.

        :param model_path: Random seed.
        """
        self.actor_model, advantage, old_prediction = self._create_actor_model()
        if os.path.exists('./' + model_path + '/PPO_actor_model.h5'):
            self.actor_model.load_weights('./' + model_path + '/PPO_actor_model.h5')
        self.actor_model.compile(
            loss=self.proximal_policy_optimization_loss(advantage=advantage, old_prediction=old_prediction),
            metrics=['mse'], optimizer=Adam(lr=self.learning_rate)
        )

        self.critic_model, _ = self._create_critic_model()
        if os.path.exists('./' + model_path + '/PPO_critic_model.h5'):
            self.critic_model.load_weights('./' + model_path + '/PPO_critic_model.h5')
        self.critic_model.compile(loss='mse', metrics=['mse'], optimizer=Adam(lr=self.learning_rate))

    def increment_step(self):
        """
        Increment the step count of the trainer
        """
        self.steps = self.steps + 1

    def update_last_reward(self, reward):
        """
        Updates the last reward
        """
        self.last_reward = reward

    def take_action(self, observation, _env):
        """
        Decides actions given state/observation information, and takes them in environment.
        :param observation: The BrainInfo from environment.
        :param _env: The environment.
        :return: the action array and an object as cookie
        """
        obs = observation.reshape(1, self.state_size)
        p = self.actor_model.predict([obs, self.dummy_advantage, self.dummy_old_prediction])
        self.last_prediction = p[0]
        if self.is_training is True:
            action = np.random.choice(self.action_size, p=np.nan_to_num(p[0]))
            return action
        else:
            action = np.argmax(p[0])
            return action

    def take_action_continous(self, observation, _env):
        obs = observation.reshape(1, self.state_size)
        p = self.actor_model.predict([obs, self.dummy_advantage, self.dummy_old_prediction])
        self.last_prediction = p[0]
        if self.is_training is True:
            action = p[0] + np.random.normal(loc=0, scale=self.exploration_noise, size=p[0].shape)
        else:
            action = p[0]
        return action

    def add_experiences(self, observation, action, next_observation, reward, done, info):
        """
        Adds experiences to each agent's experience history.
        :param observation: the observation before executing the action
        :param action: Current executed action
        :param next_observation: the observation after executing the action
        :param reward: the reward obtained after executing the action.
        :param done: true if the episode ended.
        :param info: info after executing the action.
        """
        self.replay_memory.append(
            [observation, action, next_observation, reward, done, info, self.last_prediction, reward])

    def process_experiences(self, current_info, action_vector, next_info):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param current_info: Current BrainInfo.
        :param action_vector: Current executed action
        :param next_info: Next corresponding BrainInfo.
        """
        # Update the reward for the last round if the round ends
        _len = len(self.replay_memory)
        done = self.replay_memory[_len-1][4]
        if done is True:
            if _len > 1:
                for j in range(_len - 2, -1, -1):
                    [state, action, next_state, reward, done, info, last_pred, pro_reward] = self.replay_memory[j]
                    if (done is True) and (pro_reward is not None):
                        break
                    else:
                        self.replay_memory[j][7] += self.replay_memory[j + 1][7] * self.gamma
        pass

    def end_episode(self):
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets.
        """
        # print("End Episode...")

    def is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to wether or not update_model() can be run
        """
        # The NN is ready to be updated if there is at least a batch in the replay memory
        # return (len(self.replay_memory) >= self.batch_size) and (len(self.replay_memory) % self.batch_size == 0)

        # The NN is ready to be updated everytime a batch is sampled
        return (self.steps > 1) and ((self.steps % self.batch_size) == 0)

    def transform_reward(self, reward_list):
        _list = reward_list.copy()
        if len(_list) > 1:
            for j in range(len(_list) - 2, -1, -1):
                _list[j] += _list[j + 1] * self.gamma
        return _list

    def update_model(self):
        """
        Uses the memory to update model. Run back propagation.
        """
        num_samples = min(self.batch_size, len(self.replay_memory))
        mini_batch = random.sample(self.replay_memory, num_samples)

        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch = []
        reward_batch = []
        action_batch = []
        state1_batch = []
        last_prediction_batch = []

        for [state, action, next_state, reward, done, info, last_pred, proc_reward] in mini_batch:
            state0_batch.append(state)
            state1_batch.append(next_state)
            # action matrix
            action_matrix = np.zeros(self.action_size)
            action_matrix[action] = 1
            action_batch.append(action_matrix)
            # last_prediction-probality
            last_prediction_batch.append(last_pred)
            # reward
            reward_batch.append(proc_reward)

        state0_batch = np.array(state0_batch)
        state1_batch = np.array(state1_batch)
        reward_batch = np.array(reward_batch).reshape((num_samples, 1))
        action_batch = np.array(action_batch)
        old_prediction_batch = np.array(last_prediction_batch)

        # Train actor
        pred_values_batch = np.array(self.critic_model.predict(state0_batch))
        advantage_batch = reward_batch - pred_values_batch
        actor_log = self.actor_model.train_on_batch([state0_batch, advantage_batch, old_prediction_batch], action_batch)
        train_names = ['actor_loss', 'actor_mse']
        self._write_log(self.tensorBoard, train_names, actor_log, int(self.steps / self.batch_size))

        # Train critic
        critic_log = self.critic_model.train_on_batch(state0_batch, reward_batch)
        train_names = ['critic_loss', 'critic_mse']
        self._write_log(self.tensorBoard, train_names, critic_log, int(self.steps / self.batch_size))

    def save_model(self, model_path):
        """
        Save the model architecture to i.e. Tensorboard.
        :param model_path: The path where the model will be saved.
        """
        if os.path.exists('./' + model_path):
            self.actor_model.save('./' + model_path + '/PPO_actor_model.h5')
            self.critic_model.save('./' + model_path + '/PPO_critic_model.h5')
        else:
            raise FAPSTrainerException("The model path doesn't exist. model_path : " + './' + model_path)

    def write_summary(self):
        """
        Saves training statistics to i.e. Tensorboard.
        """
        # print(self.model.summary())
        pass

    def write_tensor_board_text(self, key, input_dict):
        """
        Saves text to Tensorboard.
        Note: Only works on tensorflow r1.2 or above.
        :param key: The name of the text.
        :param input_dict: A dictionary that will be displayed in a table on Tensorboard.
        """
        try:
            s_op = tf.summary.text(key,
                                   ([[str(x), str(input_dict[x])] for x in input_dict])
                                   )
            self.tensorBoard.writer.add_summary(s_op, int(self.steps / self.batch_size))
        except:
            logger.info("Cannot write text summary for Tensorboard. Tensorflow version must be r1.2 or above.")

    def write_tensorboard_value(self, key, value):
        """
        Saves text to Tensorboard.
        Note: Only works on tensorflow r1.2 or above.
        :param key: The name of the text.
        :param value: A value that will bw displayed on Tensorboard.
        """
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = key
        self.tensorBoard.writer.add_summary(summary, int(self.steps / self.batch_size))
        self.tensorBoard.writer.flush()

    @staticmethod
    def _write_log(callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()


pass
