# # FAPS PLMAgents
import os
import random
import logging
import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as k

from FAPSPLMAgents.exception import FAPSPLMEnvironmentException
import FAPSPLMAgents.communicatorapi_python.action_type_proto_pb2 as action__type__proto__pb2

logger = logging.getLogger("FAPSPLMAgents")


class FAPSTrainerException(FAPSPLMEnvironmentException):
    """
    Related to errors with the Trainer.
    """
    pass


class DDQN:
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

        # initialize global trainer parameters
        self.brain_name = brain_name
        self.env_brain = env
        self.trainer_parameters = trainer_parameters
        self.is_training = training
        self.seed = seed
        self.steps = 0
        self.last_reward = 0
        self.initialized = False

        # initialize specific DQN parameters
        self.env_brain = env
        self.state_size = env.stateSize
        self.action_size = env.actionSize
        self.action_space_type = env.actionSpaceType
        if self.action_space_type == action__type__proto__pb2.action_continuous:
            logger.warning("Using DDQN with continuous action space. Please check your environment definition")
        self.num_layers = self.trainer_parameters['num_layers']
        self.batch_size = self.trainer_parameters['batch_size']
        self.hidden_units = self.trainer_parameters['hidden_units']
        self.replay_memory = deque(maxlen=self.trainer_parameters['memory_size'])
        self.gamma = self.trainer_parameters['gamma']  # discount rate
        self.epsilon = self.trainer_parameters['epsilon']  # exploration rate
        self.epsilon_min = self.trainer_parameters['epsilon_min']
        self.epsilon_decay = self.trainer_parameters['epsilon_decay']
        self.alpha = self.trainer_parameters['alpha']
        self.alpha_decay = self.trainer_parameters['alpha_decay']
        self.alpha_min = self.trainer_parameters['alpha_min']
        self.learning_rate = self.trainer_parameters['learning_rate']
        self.model = None
        self.target_model = None

    def __str__(self):
        return '''Double DQN Trainer'''

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

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.hidden_units, input_dim=self.state_size, activation='relu'))
        for x in range(1, self.num_layers):
            model.add(Dense(self.hidden_units, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        return model

    def _update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def is_initialized(self):
        """
        check if the trainer is initialized
        """
        return self.initialized

    def initialize(self):
        """
        Initialize the trainer
        """
        self.model = self._build_model()
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        print('\n##### Model ')
        print(self.model.summary())

        self.target_model = self._build_model()
        self.target_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        print('\n##### Target Model ')
        print(self.target_model.summary())

        self._update_target_model()
        self.initialized = True

    def clear(self):
        """
        Clear the trainer
        """
        k.clear_session()
        self.replay_memory.clear()
        self.model = None
        self.target_model = None

    def load_model_and_restore(self, model_path):
        """
        Load and restore the model from a defined path.

        :param model_path: saved model.
        """
        self.model = self._build_model()
        if os.path.exists(model_path + '/DDQN_model.h5'):
            self.model.load_weights(model_path + '/DDQN_model.h5')
            self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        else:
            self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        self.target_model = self._build_model()
        if os.path.exists(model_path + '/DDQN_target.h5'):
            self.target_model.load_weights(model_path + '/DDQN_target.h5')
            self.target_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        else:
            self.target_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def increment_step(self):
        """
        Increment the step count of the trainer
        """
        self.steps = self.steps + 1

    def update_last_reward(self, rewards):
        """
        Updates the last reward
        """
        self.last_reward = rewards

    def take_action(self, brain_info):
        """
        Decides actions given state/observation information, and takes them in environment.
        :param brain_info: The BrainInfo from environment.
        :return: the action array and an object to be passed to add experiences
        """
        # take_action_vector[brain_name], action_cookie = trainer.take_action(curr_info, self.env)

        if np.random.rand() <= self.epsilon:
            # action_cookie = random.randrange(self.action_size)
            return np.random.randint(0, 1, self.action_size)
            # return random.randrange(self.action_size)
        else:
            act_values = self.model.predict(brain_info.states)
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
        # Nothing to be done in the DQN case

    def end_episode(self):
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets.
        """
        self.replay_memory.clear()
        print("End Episode...")

    def is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to wether or not update_model() can be run
        """
        # The NN is ready to be updated everytime a batch is sampled
        return (self.steps > 1) and ((self.steps % self.batch_size) == 0)

    def update_model(self):
        """
        Uses the memory to update model. Run back propagation.
        """
        # TODO: update to support multiple agents. Now only one agent is supported
        num_samples = min(self.batch_size, len(self.replay_memory))
        mini_batch = random.sample(self.replay_memory, num_samples)

        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for state, action, reward, next_state, done in mini_batch:
            state0_batch.append(state[0])
            state1_batch.append(next_state[0])
            reward_batch.append(reward[0])
            action_batch.append(action)
            terminal1_batch.append(0. if done[0] else 1.)

        state0_batch = np.array(state0_batch)
        state1_batch = np.array(state1_batch)
        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        action_batch = np.array(action_batch)

        next_target = self.target_model.predict_on_batch(state1_batch)
        discounted_reward_batch = self.gamma * np.amax(next_target, axis=1)
        discounted_reward_batch = discounted_reward_batch * terminal1_batch
        delta_targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

        target_f = self.model.predict_on_batch(state0_batch)
        indexes = np.argmax(action_batch, axis=1)
        target_f_after = target_f
        target_f_after[:, indexes] = delta_targets

        # train the model network
        self.model.train_on_batch(state0_batch, target_f_after)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update the target network
        if self.get_step % (32 * self.batch_size):
            self._update_target_model()

        # for state, action, reward, next_state, done in mini_batch:
        #     target_f = self.model.predict(state)
        #     if done:
        #         target_f[0][np.argmax(action)] = reward[0]
        #     else:
        #         target_f[0][np.argmax(action)] = (reward + self.gamma *
        #                   np.amax(self.target_model.predict(next_state)[0]))[0]
        #     # target_f[0][action] = target
        #     self.model.fit(state, target_f, epochs=1, verbose=0)

    def save_model(self, model_path):
        """
        Save the model architecture.
        :param model_path: The path where the model will be saved.
        """
        if os.path.exists(model_path):
            self.model.save(model_path + '/DDQN_model.h5')
            self.target_model.save(model_path + '/DDQN_target.h5')
        else:
            raise FAPSTrainerException("The model path doesn't exist. model_path : " + model_path)

    def write_summary(self):
        """
        Saves training statistics to i.e. Tensorboard.
        """
        # TODO: Add Tensorboard support - Jupiter
        # print(self.model.summary())

    def write_tensorboard_text(self, key, input_dict):
        """
        Saves text to Tensorboard.
        Note: Only works on tensorflow r1.2 or above.
        :param key: The name of the text.
        :param input_dict: A dictionary that will be displayed in a table on Tensorboard.
        """
        # TODO: Add Tensorboard support
        # try:
        #     s_op = tf.summary.text(key,
        #                            tf.convert_to_tensor(([[str(x), str(input_dict[x])] for x in input_dict]))
        #                            )
        #     s = self.sess.run(s_op)
        #     self.summary_writer.add_summary(s, self.get_step)
        # except:
        #     logger.info("Cannot write text summary for Tensorboard. Tensorflow version must be r1.2 or above.")

        # print("Key: " + key + " - Value: " + input_dict)


pass
