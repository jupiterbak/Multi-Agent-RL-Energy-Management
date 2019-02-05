# # FAPS PLMAgents

import logging
import os
import random
from collections import deque

import keras.backend as k
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from FAPSPLMAgents.exception import FAPSPLMEnvironmentException

logger = logging.getLogger("FAPSPLMAgents")


class FAPSTrainerException(FAPSPLMEnvironmentException):
    """
    Related to errors with the Trainer.
    """
    pass


class A2C(object):
    """This class is the abstract class for the unitytrainers"""

    def __init__(self, env, brain_name, trainer_parameters, training, seed):
        """
        Responsible for collecting experiences and training a neural network model.

        :param env: The FAPSPLMEnvironment.
        :param brain_name: The brain to train.
        :param trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        :param seed: Random seed.
        """
        self.brain_name = brain_name
        self.brain = env
        self.trainer_parameters = trainer_parameters
        self.is_training = training
        self.seed = seed
        self.steps = 0
        self.last_reward = 0
        self.initialized = False

        # initialize specific PPO parameters
        self.env_brain = env
        self.state_size = env.stateSize
        self.action_size = env.actionSize
        self.action_space_type = env.actionSpaceType
        self.num_layers = self.trainer_parameters['num_layers']
        self.batch_size = self.trainer_parameters['batch_size']
        self.hidden_units = self.trainer_parameters['hidden_units']
        self.replay_memory = deque(maxlen=self.trainer_parameters['memory_size'])
        self.gamma = self.trainer_parameters['gamma']  # discount rate
        self.epsilon = self.trainer_parameters['epsilon']  # exploration rate
        self.epsilon_min = self.trainer_parameters['epsilon_min']
        self.epsilon_decay = self.trainer_parameters['epsilon_decay']
        self.learning_rate = self.trainer_parameters['learning_rate']
        self.actor_model = None
        self.critic_model = None

    def __str__(self):
        return '''A2C(Advantage Actor-Critic) Trainer'''

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
        return self.trainer_parameters['max_steps']

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

    def is_initialized(self):
        """
        check if the trainer is initialized
        """
        return self.initialized

    def _create_actor_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_units, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        for x in range(1, self.num_layers):
            model.add(Dense(self.hidden_units, activation='relu', kernel_initializer='he_uniform'))

        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
        return model

    def _create_critic_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_units, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        for x in range(1, self.num_layers):
            model.add(Dense(self.hidden_units,  activation='relu', kernel_initializer='he_uniform'))

        model.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))
        return model

    def initialize(self):
        """
        Initialize the trainer
        """
        self.actor_model = self._create_actor_model()
        self.actor_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
        print('\n##### Actor Model ')
        print(self.actor_model.summary())

        self.critic_model = self._create_critic_model()
        self.critic_model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        print('\n##### Critic Model ')
        print(self.critic_model.summary())
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
        self.actor_model = self._create_actor_model()
        if os.path.exists(model_path + '/A2C_actor_model.h5'):
            self.actor_model.load_weights(model_path + '/A2C_actor_model.h5')
            self.actor_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
        else:
            self.actor_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))

        self.critic_model = self._create_critic_model()
        if os.path.exists(model_path + '/A2C_critic_model.h5'):
            self.critic_model.load_weights(model_path + '/A2C_critic_model.h5')
            self.critic_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        else:
            self.critic_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

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

    def take_action(self, brain_info):
        """
        Decides actions given state/observation information, and takes them in environment.
        :param brain_info: The BrainInfo from environment.
        :return: the action array and an object to be passed to add experiences
        """
        policy = self.actor_model.predict(brain_info.states, batch_size=1).flatten()
        index = np.random.choice(self.action_size, 1, p=policy)[0]
        rslt = np.zeros(shape=self.action_size, dtype=np.dtype(int))
        rslt[index] = 1
        #print("Hello !!!!")
        return rslt

    def add_experiences(self, curr_info, action_vector, next_info):
        """
        Adds experiences to each agent's experience history.
        :param action_vector: Current executed action
        :param curr_info: Current AllBrainInfo.
        :param next_info: Next AllBrainInfo.
        """
        self.replay_memory.append(
            (curr_info.states, action_vector, [next_info.rewards], next_info.states, [next_info.local_done]))

    def process_experiences(self, current_info, action_vector, next_info):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param current_info: Current BrainInfo.
        :param action_vector: Current executed action
        :param next_info: Next corresponding BrainInfo.
        """
        # Nothing to be done.

    def end_episode(self):
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets.
        """
        self.replay_memory.clear()

    def is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to wether or not update_model() can be run
        """
        # The NN is ready to be updated everytime a batch is sampled
        return (self.steps > 1) and ((self.steps % self.batch_size) == 0)

    def update_model(self):
        """
        Uses training_buffer to update model. Run back propagation.
        """
        num_samples = min(self.batch_size, len(self.replay_memory))
        mini_batch = random.sample(self.replay_memory, num_samples)

        states = np.zeros((self.batch_size, self.state_size))
        advantagess = np.zeros((self.batch_size, self.action_size))
        targets = np.zeros((self.batch_size, 1))
        i = 0

        for state, action, reward, next_state, done in mini_batch:
            target = np.zeros((1, 1))
            advantages = np.zeros((1, self.action_size))

            value = self.critic_model.predict(state)[0]
            next_value = self.critic_model.predict(next_state)[0]

            if done:
                advantages[0][np.argmax(action)] = reward - value
                target[0] = reward
            else:
                advantages[0][np.argmax(action)] = reward + self.gamma * next_value - value
                target[0] = reward + self.gamma * next_value

            states[i] = state
            targets[i] = target
            advantagess[i] = advantages
            i = i+1

        self.actor_model.fit(states, advantagess, epochs=1, verbose=0)
        self.critic_model.fit(states, targets, epochs=1, verbose=0)

    def save_model(self, model_path):
        """
        Save the model architecture to i.e. Tensorboard.
        :param model_path: The path where the model will be saved.
        """
        if os.path.exists(model_path):
            self.actor_model.save(model_path + '/A2C_actor_model.h5')
            self.critic_model.save(model_path + '/A2C_critic_model.h5')
        else:
            raise FAPSTrainerException("The model path doesn't exist. model_path : " + model_path)

    def write_summary(self):
        """
        Saves training statistics to i.e. Tensorboard.
        """
        # Nothing to be done.

    def write_tensorboard_text(self, key, input_dict):
        """
        Saves text to Tensorboard.
        Note: Only works on tensorflow r1.2 or above.
        :param key: The name of the text.
        :param input_dict: A dictionary that will be displayed in a table on Tensorboard.
        """
        # Nothing to be done.


pass
