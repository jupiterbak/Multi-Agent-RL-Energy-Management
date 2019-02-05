# # FAPS PLMAgents
# ## FAPS PLM ML-Agent Learning

import logging
import numpy as np
import os
import re
import tensorflow as tf
import keras
import yaml
import keras.backend as backend
import gym
import gym_eflex_agent
from OpenAIGym.FAPSPLMAgents import FAPSPLMEnvironmentException


class TrainerController(object):
    def __init__(self, use_gpu, brain_name, environment, save_freq, load, train, keep_checkpoints, lesson, seed,
                 trainer_config_path):
        """
        :param brain_name: Name of the brain to train
        :param save_freq: Frequency at which to save model
        :param load: Whether to load the model or randomly initialize
        :param train: Whether to train model, or only run inference
        :param environment: Environment to user
        :param keep_checkpoints: How many model checkpoints to keep
        :param lesson: Start learning from this lesson
        :param seed: Random seed used for training
        :param trainer_config_path: Fully qualified path to location of trainer configuration file
        """

        self.use_gpu = use_gpu
        self.trainer_config_path = trainer_config_path
        self.logger = logging.getLogger("FAPSPLMAgents")
        self.environment = environment
        self.save_freq = save_freq
        self.lesson = lesson
        self.load_model = load
        self.train_model = train
        self.keep_checkpoints = keep_checkpoints
        self.trainers = {}
        if seed == -1:
            seed = np.random.randint(0, 999999)
        self.seed = seed
        np.random.seed(self.seed)

        if backend.backend() == 'tensorflow':
            tf.set_random_seed(self.seed)
        else:
            np.random.seed(seed)

        self.env = gym.make(environment)
        self.env.seed(self.seed)

        # Reset the environement and get all parameters from the simulation
        self.all_info = self.env.reset(train_mode=self.train_model)
        self.brain_name = re.sub('[^0-9a-zA-Z]+', '-', brain_name)
        self.model_path = 'models/%s' % self.brain_name

    def _get_progress(self, brain_name, step_progress, reward_progress):
        """
        Compute and increment the progess of a specified trainer.
        :param brain_name: Name of the brain to train
        :param step_progress: last step
        :param reward_progress: last cummulated reward
        """
        step_progress += self.trainers[brain_name].get_step / self.trainers[brain_name].get_max_steps
        reward_progress += self.trainers[brain_name].get_last_reward
        return step_progress, reward_progress

    def _save_model(self):
        """
        Saves current model to checkpoint folder.
        """
        for k, t in self.trainers.items():
            t.save_model(self.model_path)
        print("\nINFO: Model saved.")

    @staticmethod
    def _import_module(module_name, class_name):
        """Constructor"""
        module = __import__(module_name)
        my_class = getattr(module, class_name)
        my_class = getattr(my_class, class_name)
        return my_class

    def _initialize_trainer(self, brain_name, trainer_config):
        self.trainer_parameters_dict = {}
        self.trainers = {}

        trainer_parameters = trainer_config['default'].copy()
        graph_scope = re.sub('[^0-9a-zA-Z]+', '-', brain_name)
        trainer_parameters['graph_scope'] = graph_scope
        trainer_parameters['summary_path'] = '{basedir}/{name}'.format(
            basedir='summaries',
            name=str(self.run_id) + '_' + graph_scope)
        if brain_name in trainer_config:
            _brain_key = brain_name
            while not isinstance(trainer_config[_brain_key], dict):
                _brain_key = trainer_config[_brain_key]
            for k in trainer_config[_brain_key]:
                trainer_parameters[k] = trainer_config[_brain_key][k]
        self.trainer_parameters_dict[brain_name] = trainer_parameters.copy()

        # Instantiate the trainer
        # import the module
        module_spec = self._import_module("FAPSPLMAgents." + self.trainer_parameters_dict[brain_name]['trainer'],
                                          self.trainer_parameters_dict[brain_name]['trainer'])
        if module_spec is None:
            raise FAPSPLMEnvironmentException("The trainer config contains an unknown trainer type for brain {}"
                                              .format(brain_name))
        else:
            self.trainers[brain_name] = module_spec(self.env, brain_name, self.trainer_parameters_dict[brain_name],
                                                    self.train_model, self.seed)

    def _load_config(self):
        try:
            with open(self.trainer_config_path) as data_file:
                trainer_config = yaml.load(data_file)
                return trainer_config
        except IOError:
            raise FAPSPLMEnvironmentException("""Parameter file could not be found here {}.
                                            Will use default Hyper parameters"""
                                              .format(self.trainer_config_path))
        except UnicodeDecodeError:
            raise FAPSPLMEnvironmentException("There was an error decoding Trainer Config from this path : {}"
                                              .format(self.trainer_config_path))

    @staticmethod
    def _create_model_path(model_path):
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except Exception:
            raise FAPSPLMEnvironmentException("The folder {} containing the generated model could not be accessed. "
                                              "Please make sure the permissions are set correctly.".format(model_path))

    def start_learning(self):

        # configure tensor flow to use 8 cores
        if self.use_gpu:
            if backend.backend() == 'tensorflow':
                config = tf.ConfigProto(device_count={"GPU": 1},
                                        intra_op_parallelism_threads=8,
                                        inter_op_parallelism_threads=8,
                                        allow_soft_placement=True)
                keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
            else:
                raise FAPSPLMEnvironmentException("Other backend environment than Tensorflow are nor supported. ")
        else:
            if backend.backend() == 'tensorflow':
                config = tf.ConfigProto(device_count={"CPU": 8},
                                        intra_op_parallelism_threads=8,
                                        inter_op_parallelism_threads=8,
                                        allow_soft_placement=True)
                keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
            else:
                raise FAPSPLMEnvironmentException("Other backend environment than Tensorflow are nor supported. ")

        trainer_config = self._load_config()
        self._create_model_path(self.model_path)

        # Choose and instantiate a trainer
        # TODO: Instantiate the trainer depending on the configurations sent by the
        # environment. Current implementation considers only the brain started on the server side - Jupiter
        self._initialize_trainer(self.brain_name, trainer_config)

        print("\n##################################################################################################")
        print("Starting Training...")
        print("Brain Name: {}".format(self.brain_name))
        print("Backend : {}".format(backend.backend()))
        print("Use cpu: {}".format(self.use_gpu))
        iterator = 0
        for k, t in self.trainers.items():
            print("Trainer({}): {}".format(iterator, t.__str__()))
            iterator = iterator + 1
        print("##################################################################################################")

        # Initialize the trainer
        for k, t in self.trainers.items():
            t.initialize()

        # Instantiate model parameters
        if self.load_model:
            print("\nINFO: Loading models ...")
            for k, t in self.trainers.items():
                t.load_model_and_restore(self.model_path)

        global_step = 0  # This is only for saving the model
        curr_info = self.all_info[self.brain_name]

        if self.train_model:
            for brain_name, trainer in self.trainers.items():
                trainer.write_tensorboard_text('Hyperparameters', trainer.parameters)
        try:
            while any([t.get_step <= t.get_max_steps for k, t in self.trainers.items()]) or not self.train_model:
                # reset if global_done or end of an episode of the trainer
                if self.env.global_done or any([d != 0 for d in curr_info.local_done]):
                    curr_info = self.env.reset()
                    for brain_name, trainer in self.trainers.items():
                        trainer.end_episode()

                # Decide and take an action
                new_info = 0
                action_vector = {}

                for brain_name, trainer in self.trainers.items():
                    action_vector[brain_name] = trainer.take_action(curr_info)
                    new_info = self.env.step(action_vector[brain_name])[brain_name]
                    # TODO: find a better implementation - Jupiter

                # Process experience and generate statistics
                for brain_name, trainer in self.trainers.items():
                    trainer.add_experiences(curr_info, action_vector[brain_name], new_info)
                    trainer.process_experiences(curr_info, action_vector[brain_name], new_info)
                    if trainer.is_ready_update() and self.train_model and trainer.get_step <= trainer.get_max_steps:
                        # Perform gradient descent with experience buffer
                        trainer.update_model()
                        # Write training statistics.
                        trainer.write_summary()
                    if self.train_model and trainer.get_step <= trainer.get_max_steps:
                        trainer.increment_step()
                        trainer.update_last_reward(new_info.rewards)

                # Update Global Step
                if self.train_model and trainer.get_step <= trainer.get_max_steps:
                    global_step += 1

                # Save the model by the save frequency
                if global_step % self.save_freq == 0 and global_step != 0 and self.train_model:
                    # Save model
                    self._save_model()
                curr_info = new_info

            # Final save  model
            if global_step != 0 and self.train_model:
                self._save_model()

        except KeyboardInterrupt:
            if self.train_model:
                self.logger.info("Learning was interrupted. Please wait while the graph is generated.")
                self._save_model()
            pass

        # Clear the trainer
        for k, t in self.trainers.items():
            t.clear()

        # clear the backend
        backend.clear_session()

        self.env.close()  # If needed save some parameters

        print("\n##################################################################################################")
        print("Training ended. ")
        print("##################################################################################################")
