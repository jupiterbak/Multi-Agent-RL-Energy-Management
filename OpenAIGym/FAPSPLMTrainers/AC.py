# # FAPS PLMAgents

import logging
import numpy as np
import os
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
from collections import deque

from FAPSPLMAgents.exception import FAPSPLMEnvironmentException, BrainInfo

logger = logging.getLogger("FAPSPLMAgents")


class FAPSTrainerException(FAPSPLMEnvironmentException):
    """
    Related to errors with the Trainer. - Not implemented
    """
    pass


class AC(object):
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
        self.target_actor_model = None
        self.state_input = None
        self.critic_model = None
        self.target_critic_model = None
        self.critic_state_input = None
        self.critic_action_input = None

    def __str__(self):
        return '''Actor Critic Trainer'''

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

    def _create_actor_model(self):
        # Neural Net for Deep-Q learning Model
        state_input = Input(shape=self.state_size)
        state_h1 = Dense(self.hidden_units, activation='relu')(state_input)
        for x in range(1, self.num_layers -1):
            state_h1 = Dense(self.hidden_units, activation='relu')(state_h1)
        output = Dense(self.action_size, activation='relu')(state_h1)
        model = Model(input=state_input, output=output)
        return state_input, model

    def _create_critic_model(self):
        state_input = Input(shape=self.state_size)
        state_h1 = Dense(self.hidden_units, activation='relu')(state_input)
        state_h2 = Dense(2*self.hidden_units)(state_h1)

        action_input = Input(shape=self.action_size)
        action_h1 = Dense(2*self.hidden_units)(action_input)

        merged = Add()([state_h2, action_h1])
        for x in range(1, self.num_layers):
            merged = Dense(self.hidden_units, activation='relu')(merged)

        output = Dense(1, activation='relu')(merged)
        model = Model(input=[state_input, action_input], output=output)
        return state_input, action_input, model

    def is_initialized(self):
        """
        check if the trainer is initialized
        """
        return self.initialized

    def initialize(self):
        """
        Initialize the trainer
        """
        self.state_input, self.actor_model = self._create_actor_model()
        self.actor_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        print(self.actor_model.summary())

        _, self.target_actor_model = self._create_actor_model()
        self.target_actor_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        self.actor_critic_grad = tf.placeholder(tf.float32,
                                                [None, self.env.action_space.shape[
                                                    0]])  # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights, -self.actor_critic_grad)  # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        self.critic_state_input, self.critic_action_input, self.critic_model = self._create_critic_model()
        _, _, self.target_critic_model = self._create_critic_model()
        self.target_critic_model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

        self.initialized = True

    def clear(self):
        """
        Clear the trainer
        """
        K.clear_session()
        self.replay_memory.clear()
        self.actor_model = None
        self.critic_model = None
        self.target_actor_model = None
        self.target_critic_model = None

    def _update_actor_target_model(self):
        # copy weights from model to target_actor_model
        self.target_actor_model.set_weights(self.actor_model.get_weights())

    def _update_critic_target_model(self):
        # copy weights from model to target_critic_model
        self.target_critic_model.set_weights(self.critic_model.get_weights())

    def load_model_and_restore(self, model_path):
        """
        Load and restore the model from a defined path.

        :param model_path: Random seed.
        """
        _, self.actor_model = self._create_actor_model()
        if os.path.exists(model_path + '/AC_actor_model.h5'):
            self.actor_model.load_weights(model_path + '/AC_actor_model.h5')
            self.actor_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        else:
            self.actor_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        _, self.target_actor_model = self._create_actor_model()
        if os.path.exists(model_path + '/AC_actor_target.h5'):
            self.target_actor_model.load_weights(model_path + '/AC_actor_target.h5')
            self.target_actor_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        else:
            self.target_actor_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        _, _, self.critic_model = self._create_critic_model()
        if os.path.exists(model_path + '/AC_critic_model.h5'):
            self.critic_model.load_weights(model_path + '/AC_critic_model.h5')
            self.critic_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        else:
            self.critic_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        _, _, self.target_critic_model = self._create_critic_model()
        if os.path.exists(model_path + '/AC_critic_target.h5'):
            self.target_critic_model.load_weights(model_path + '/AC_critic_target.h5')
            self.target_critic_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        else:
            self.target_critic_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

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
        if np.random.rand() <= self.epsilon:
            # action_cookie = random.randrange(self.action_size)
            return np.random.randint(0, 1, self.action_size)
        else:
            act_values = self.actor_model.predict(brain_info.states)
            # action_cookie = np.argmax(act_values[0])
            index = np.argmax(act_values[0])
            rslt = np.zeros(shape=act_values[0].shape, dtype=np.dtype(int))
            rslt[index] = 1
            return rslt  # returns action

    def add_experiences(self, curr_info, action_vector, next_info):
        """
        Adds experiences to each agent's experience history.
        :param action_vector: Current executed action
        :param curr_info: Current AllBrainInfo.
        :param next_info: Next AllBrainInfo.
        """
        self.replay_memory.append(
            (curr_info.states, action_vector, next_info.rewards, next_info.states, next_info.local_done))

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
        # Nothing to be done.

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state, action], reward, verbose=0)
            
    def save_model(self, model_path):
        """
        Save the model architecture to i.e. Tensorboard.
        :param model_path: The path where the model will be saved.
        """
        if os.path.exists(model_path):
            self.actor_model.save(model_path + '/AC_actor_model.h5')
            self.target_actor_model.save(model_path + '/AC_actor_target.h5')
            self.critic_model.save(model_path + '/AC_critic_model.h5')
            self.target_critic_model.save(model_path + '/AC_critic_target.h5')
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
