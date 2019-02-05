# # FAPS PLMAgents
# ## FAPS PLM ML-Agent Learning

import logging

import os
import docopt
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


def _create_model_path(model_path):
    try:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    except Exception:
        raise Exception("The folder {} containing the generated model could not be accessed. "
                        "Please make sure the permissions are set correctly.".format(model_path))


def _load_config(_trainer_config_path):
    try:
        with open(_trainer_config_path) as data_file:
            trainer_config = yaml.load(data_file)
            return trainer_config
    except IOError:
        raise Exception("""Parameter file could not be found here {}.
                                        Will use default Hyper parameters""".format(_trainer_config_path))
    except UnicodeDecodeError:
        raise Exception("There was an error decoding Trainer Config from this path : {}".format(_trainer_config_path))


def _import_module(module_name, class_name):
    """Constructor"""
    module = __import__(module_name)
    my_class = getattr(module, class_name)
    my_class = getattr(my_class, class_name)
    return my_class


def _initialize_trainer(_brain_name, _trainer_config, _env, _train_model, _seed):
    trainer_parameters_dict = {}
    trainers = {}

    trainer_parameters = _trainer_config['default'].copy()
    graph_scope = re.sub('[^0-9a-zA-Z]+', '-', _brain_name)
    trainer_parameters['graph_scope'] = graph_scope

    if _brain_name in _trainer_config:
        _brain_key = _brain_name
        for k in _trainer_config[_brain_key]:
            trainer_parameters[k] = _trainer_config[_brain_key][k]
        trainer_parameters_dict[_brain_name] = trainer_parameters.copy()

    # Instantiate the trainer
    # import the module
    module_spec = _import_module("FAPSPLMAgents." + trainer_parameters_dict[_brain_name]['trainer'],
                                 trainer_parameters_dict[_brain_name]['trainer'])
    if module_spec is None:
        raise Exception("The trainer config contains an unknown trainer type for brain {}".format(_brain_name))
    else:
        trainers[brain_name] = module_spec(_env, _brain_name, trainer_parameters_dict[brain_name], _train_model, _seed)

    return trainer_parameters_dict, trainers


def start_learning(_brain_name, _use_gpu, trainer_config_path, _train_model, _load_model, _seed):
    # configure tensor flow to use 8 cores
    if _use_gpu:
        if backend.backend() == 'tensorflow':
            config = tf.ConfigProto(device_count={"GPU": 1},
                                    intra_op_parallelism_threads=8,
                                    inter_op_parallelism_threads=8,
                                    allow_soft_placement=True)
            keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
        else:
            raise Exception("Other backend environment than Tensorflow are nor supported. ")
    else:
        if backend.backend() == 'tensorflow':
            config = tf.ConfigProto(device_count={"CPU": 8},
                                    intra_op_parallelism_threads=8,
                                    inter_op_parallelism_threads=8,
                                    allow_soft_placement=True)
            keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
        else:
            raise Exception("Other backend environment than Tensorflow are nor supported. ")

    # Load the Trainer configuration
    trainer_config = _load_config(trainer_config_path)

    # Create the model path
    model_path = 'models/%s' % _brain_name
    _create_model_path(model_path)

    # Choose and instantiate a trainer
    trainer_parameters_dict, trainers = _initialize_trainer(_brain_name, trainer_config, _env,  _train_model, _seed)

    print("\n##################################################################################################")
    print("Starting Training...")
    print("Brain Name: {}".format(_brain_name))
    print("Backend : {}".format(backend.backend()))
    print("Use cpu: {}".format(_use_gpu))
    iterator = 0
    for k, t in trainers.items():
        print("Trainer({}): {}".format(iterator, t.__str__()))
        iterator = iterator + 1
    print("##################################################################################################")

    # Initialize the trainer
    for k, t in trainers.items():
        t.initialize()

    # Instantiate model parameters
    if _load_model:
        print("\nINFO: Loading models ...")
        for k, t in trainers.items():
            t.load_model_and_restore(model_path)

    global_step = 0  # This is only for saving the model
    curr_info = _env.reset()

    if _train_model:
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


if __name__ == '__main__':
    logger = logging.getLogger("FAPSPLMAgents")

    _USAGE = '''
    Usage:
      main [options]
      main --help

    Options:
      --brain_name=<path>        Name of the brain to use. [default: DQN]. 
      --environment=<env>        Name of the environment to use. [default: 'eflex-agent-v0'].
      --keep-checkpoints=<n>     How many model checkpoints to keep [default: 5].
      --lesson=<n>               Start learning from this lesson [default: 0].
      --load                     Whether to load the model or randomly initialize [default: True].
      --save-freq=<n>            Frequency at which to save model [default: 10000].
      --seed=<n>                 Random seed used for training [default: -1].
      --train                    Whether to train model, or only run inference [default: False].
      --use-gpu                  Make use of GPU.
    '''
    options = None
    try:
        options = docopt.docopt(_USAGE)

    except docopt.DocoptExit as e:
        # The DocoptExit is thrown when the args do not match.
        # We print a message to the user and the usage block.

        print('Invalid Command!')
        print(e)
        exit(1)

    # General parameters
    brain_name = options['--run-id']
    environment = int(options['--environment'])
    seed = int(options['--seed'])
    load_model = options['--load']
    train_model = options['--train']
    save_freq = int(options['--save-freq'])
    keep_checkpoints = int(options['--keep-checkpoints'])

    lesson = int(options['--lesson'])
    use_gpu = int(options['--use-gpu'])

    # log the configuration
    # logger.info(options)

    # Constants
    # Assumption that this yaml is present in same dir as this file
    base_path = os.path.dirname(__file__)
    TRAINER_CONFIG_PATH = os.path.abspath(os.path.join(base_path, "trainer_config.yaml"))

    tc = TrainerController(use_gpu, brain_name, environment, save_freq, load_model, train_model,
                           keep_checkpoints, lesson, seed, TRAINER_CONFIG_PATH)
    tc.start_learning()
    exit(0)

# import sys
# import time
# import logging
# import gym
# import gym_eflex_agent
#
#
# sys.path.insert(0, "..")
#
# try:
#     from IPython import embed
# except ImportError:
#     import code
#
#     def embed():
#         vars = globals()
#         vars.update(locals())
#         shell = code.InteractiveConsole(vars)
#         shell.interact()
#
# interactive = True
#
# if __name__ == "__main__":
#     # optional: setup logging
#     logging.basicConfig(level=logging.WARN)
#     logger = logging.getLogger("opcua.address_space")
#     logger.setLevel(logging.DEBUG)
#
#     # test gym
#     env = gym.make('eflex-agent-v0')
#     observation = env.reset()
#     for _ in range(5000):
#         observation, reward, done, info = env.step(env.action_space.sample())
#         print(info['info'])
#         env.render('human')
#     env.close()
#
#     # try:
#     #     if interactive:
#     #         embed()
#     #     else:
#     #         while True:
#     #             time.sleep(0.5)
#     #
#     # except IOError:
#     #     pass
#     # finally:
#     #     print("done")
