# # FAPS PLMAgents

import logging
import os
import random
from collections import deque

import keras.backend as k
import numpy as np
from keras import Input
from keras.initializers import RandomUniform
from keras.layers import Dense, add
from keras.models import Sequential, Model
from keras.optimizers import Adam, Optimizer

from FAPSPLMAgents.exception import FAPSPLMEnvironmentException
from FAPSPLMTrainers.utils.random import OrnsteinUhlenbeckProcess

logger = logging.getLogger("FAPSPLMAgents")


class FAPSTrainerException(FAPSPLMEnvironmentException):
    """
    Related to errors with the Trainer.
    """
    pass


class AdditionalUpdatesOptimizer(Optimizer):
    def __init__(self, optimizer, additional_updates):
        super(AdditionalUpdatesOptimizer, self).__init__()
        self.optimizer = optimizer
        self.additional_updates = additional_updates

    def get_updates(self, params, loss):
        updates = self.optimizer.get_updates(params=params, loss=loss)
        updates += self.additional_updates
        self.updates = updates
        return self.updates

    def get_config(self):
        return self.optimizer.get_config()


class DDPG(object):
    """This class is the abstract class for the faps trainers"""

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
        self.tau = self.trainer_parameters['tau']
        self.actor_model = None
        self.target_model = None
        self.critic_model = None
        self.critic_target_model = None
        self.critic_gradient_wrt_action = None  # GRADIENTS for policy update
        self.critic_gradient_wrt_action_fn = None
        self.critic_state = None
        self.critic_action = None
        self.actor_train_fn = None
        self.random_process = None

    def __str__(self):
        return '''Deep Deterministic Policy Gradient Trainer'''

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

    def _create_actor_model(self):
        a = Input(shape=[self.state_size], name='actor_state')
        h = Dense(self.hidden_units, activation='relu', kernel_initializer='he_uniform')(a)
        for x in range(1, self.num_layers):
            h = Dense(self.hidden_units, activation='relu', kernel_initializer='he_uniform')(h)
        o = Dense(self.action_size, activation='linear', kernel_initializer=RandomUniform(minval=-0.001, maxval=0.001))(h)
        model = Model(inputs=a, outputs=o)
        return model

    def _create_critic_model(self):
        s = Input(shape=[self.state_size], name='critic_state')
        a = Input(shape=[self.action_size], name='critic_action')
        w1 = Dense(self.hidden_units, activation='relu', kernel_initializer='he_uniform')(s)
        h1 = Dense(self.hidden_units, activation='linear', kernel_initializer='he_uniform')(w1)

        a1 = Dense(self.hidden_units, activation='linear')(a)
        h2 = add([h1, a1])
        for x in range(1, self.num_layers):
            h2 = Dense(self.hidden_units, activation='relu', kernel_initializer='he_uniform')(h2)

        v = Dense(1, activation='linear', kernel_initializer=RandomUniform(minval=-0.001, maxval=0.001))(h2)
        model = Model(inputs=[s, a], outputs=v)

        return model, s, a

    @staticmethod
    def _huber_loss(y_true, y_pred, clip_value):
        # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
        # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
        # for details.
        assert clip_value > 0.

        x = y_true - y_pred
        if np.isinf(clip_value):
            # Spacial case for infinity since Tensorflow does have problems
            # if we compare `K.abs(x) < np.inf`.
            return .5 * k.square(x)

        condition = k.abs(x) < clip_value
        squared_loss = .5 * k.square(x)
        linear_loss = clip_value * (k.abs(x) - .5 * clip_value)
        if k.backend() == 'tensorflow':
            import tensorflow as tf
            if hasattr(tf, 'select'):
                return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
            else:
                return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
        else:
            raise RuntimeError('Unknown backend "{}".'.format(k.backend()))
        pass

    def _clipped_error(self, y_true, y_pred, delta_clip=np.inf):
        return k.mean(self._huber_loss(y_true, y_pred, delta_clip), axis=-1)

    def is_initialized(self):
        """
        check if the trainer is initialized
        """
        return self.initialized

    def initialize(self):
        """
        Initialize the trainer
        """
        self.actor_model = self._create_actor_model()
        self.actor_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        print('\n##### Actor Model ')
        print(self.actor_model.summary())

        self.target_model = self._create_actor_model()
        self.target_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        print('\n##### Actor Target Model ')
        print(self.target_model.summary())

        self.critic_target_model, target_critic_state_input, target_critic_action_input = self._create_critic_model()
        self.critic_target_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        print('\n##### Critic Target Model ')
        print(self.critic_target_model.summary())

        self.critic_model, self.critic_state, self.critic_action = self._create_critic_model()
        # compile the critic using 'mse' lost
        # self.critic_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        # ### Alternative with huber loss
        self.critic_model.compile(loss=self._clipped_error, optimizer=Adam(lr=self.learning_rate))
        # ### Alternative including with target updates
        # critic_updates = self._get_soft_target_model_updates(self.critic_target_model, self.critic_model, self.tau)
        # critic_optimizer = AdditionalUpdatesOptimizer(Adam(lr=self.learning_rate), critic_updates)
        # self.critic_model.compile(optimizer=critic_optimizer, loss=self._clipped_error, metrics='mse')
        print('\n##### Critic Model ')
        print(self.critic_model.summary())

        # Combine actor and critic so that we can get the policy gradient.
        # Assuming critic's state inputs are the same as actor's.
        critic_state_inputs = [self.critic_state]
        combined_critic_inputs = [self.critic_state, self.actor_model(critic_state_inputs)]
        combined_critic_output = self.critic_model(combined_critic_inputs)

        # get update with regards to the critic outputs
        updates = self.actor_model.optimizer.get_updates(params=self.actor_model.trainable_weights,
                                                         loss=-k.mean(combined_critic_output))
        # include other updates of the actor, e.g. for BN
        updates += self.actor_model.updates

        # Finally, combine it all into a callable function.
        if k.backend() == 'tensorflow':
            self.actor_train_fn = k.function(critic_state_inputs,
                                             [self.actor_model(critic_state_inputs)], updates=updates)
        else:
            self.actor_train_fn = k.function(critic_state_inputs,
                                             [self.actor_model(critic_state_inputs)], updates=updates)

        self.random_process = OrnsteinUhlenbeckProcess(size=self.action_size, theta=.15, mu=0., sigma=.2)

        # # Set the gradient for policy update
        # self.critic_gradient_wrt_action = k.gradients(combined_critic_output, critic_state_inputs)[0]
        # # Finally, combine it into a callable function.
        # if k.backend() == 'tensorflow':
        #     self.critic_gradient_wrt_action_fn = k.function(critic_state_inputs, [self.critic_gradient_wrt_action])
        # else:
        #     raise RuntimeError('Unknown backend "{}".'.format(k.backend()))

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
        # self.actor_model = self._create_actor_model()
        if os.path.exists(model_path + '/DPPG_actor_model.h5'):
            self.actor_model.load_weights(model_path + '/DPPG_actor_model.h5')
        self.actor_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        # self.target_model = self._create_actor_model()
        if os.path.exists(model_path + '/DPPG_actor_target_model.h5'):
            self.target_model.load_weights(model_path + '/DPPG_actor_target_model.h5')
        self.target_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        # self.critic_target_model, target_critic_state_input, target_critic_action_input = self._create_critic_model()
        if os.path.exists(model_path + '/DPPG_critic_target_model.h5'):
            self.critic_target_model.load_weights(model_path + '/DPPG_critic_target_model.h5')
        self.critic_target_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        # self.critic_model, self.critic_state, self.critic_action = self._create_critic_model()
        if os.path.exists(model_path + '/DPPG_critic_model.h5'):
            self.critic_model.load_weights(model_path + '/DPPG_critic_model.h5')

        # Compile the model using 'mse' lost
        # self.critic_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        # alternative with huber loss
        self.critic_model.compile(loss=self._clipped_error, optimizer=Adam(lr=self.learning_rate))
        # Alternative with target updates
        # critic_updates = self._get_soft_target_model_updates(self.critic_target_model, self.critic_model,
        # self.tau)
        # critic_optimizer = AdditionalUpdatesOptimizer(Adam(lr=self.learning_rate), critic_updates)
        # self.critic_model.compile(optimizer=critic_optimizer, loss=self._clipped_error, metrics='mse')

        # # Combine actor and critic so that we can get the policy gradient.
        # # Assuming critic's state inputs are the same as actor's.
        # critic_state_inputs = [self.critic_state]
        # combined_critic_inputs = [self.critic_state, self.actor_model(critic_state_inputs)]
        # combined_critic_output = self.critic_model(combined_critic_inputs)
        #
        # # get update with regards to the critic outputs
        # updates = self.actor_model.optimizer.get_updates(params=self.actor_model.trainable_weights,
        #                                                  loss=-k.mean(combined_critic_output))
        # # include other updates of the actor, e.g. for BN
        # updates += self.actor_model.updates
        #
        # # Finally, combine it all into a callable function.
        # if k.backend() == 'tensorflow':
        #     self.actor_train_fn = k.function(critic_state_inputs,
        #                                      [self.actor_model(critic_state_inputs)], updates=updates)
        # else:
        #     self.actor_train_fn = k.function(critic_state_inputs,
        #                                      [self.actor_model(critic_state_inputs)], updates=updates)

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
        action = self.actor_model.predict(brain_info.states, batch_size=1).flatten()
        # Apply noise, if we are in training.
        if self.is_training:
            noise = self.random_process.sample()
            action += noise

        return action

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
        # self.replay_memory.clear()
        print("\nINFO: Next episode.")

    def is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to wether or not update_model() can be run
        """
        # The NN is ready to be updated everytime a batch is sampled
        return (self.steps > 1) and ((self.steps % self.batch_size) == 0)

    def _copy_target_models(self):
        critic_weights = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)

        actor_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def update_model(self):
        """
        Uses training_buffer to update model. Run back propagation.
        """
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
            reward_batch.append(reward)
            action_batch.append(action)
            terminal1_batch.append([0.] if done[0] else [1.])

        state0_batch = np.array(state0_batch)
        state1_batch = np.array(state1_batch)
        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        action_batch = np.array(action_batch)

        # Update critic
        target_actions = self.target_model.predict_on_batch(state1_batch)
        target_q_values = self.critic_target_model.predict_on_batch([state1_batch, target_actions])
        discounted_reward_batch = self.gamma * target_q_values
        discounted_reward_batch = discounted_reward_batch * terminal1_batch
        added = reward_batch + discounted_reward_batch
        targets = added.reshape(self.batch_size, 1)
        self.critic_model.train_on_batch([state0_batch, action_batch], targets)

        # Update actor
        action_values = self.actor_train_fn([state0_batch])[0]
        assert action_values.shape == (self.batch_size, self.action_size)

        # Copy target model
        self._copy_target_models()

        # Get the gradient of the critic with respect to the actions
        # a_for_grad = self.actor_model.predict_on_batch(state0_batch)
        # critic_grad = self.critic_gradient_wrt_action_fn([state0_batch])

        print("Execute Minibatch Gradient descend: step = {} - reward  = {} - advantage = {}".format(
            self.steps, reward_batch.mean(), target_q_values.mean()))

    def save_model(self, model_path):
        """
        Save the model architecture to i.e. Tensorboard.
        :param model_path: The path where the model will be saved.
        """
        if os.path.exists(model_path):
            self.actor_model.save(model_path + '/DPPG_actor_model.h5')
            self.target_model.save(model_path + '/DPPG_actor_target_model.h5')
            self.critic_model.save(model_path + '/DPPG_critic_model.h5')
            self.critic_target_model.save(model_path + '/DPPG_critic_target_model.h5')
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
