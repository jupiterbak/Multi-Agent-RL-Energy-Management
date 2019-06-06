# # FAPS PLMAgents
import logging

from FAPSPLMAgents import FAPSPLMEnvironmentException

logger = logging.getLogger("FAPSPLMAgents")


class FAPSTrainerException(FAPSPLMEnvironmentException):
    """
    Related to errors with the Trainer.
    """
    pass


class Trainer(object):
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
        self.env_brain = env
        self.state_size = env.stateSize
        self.action_size = env.actionSize
        self.action_space_type = env.actionSpaceType
        self.trainer_parameters = trainer_parameters
        self.is_training = training
        self.seed = seed
        self.steps = 0
        self.last_reward = 0
        self.initialized = False
        self.global_done = False

    def __str__(self):
        return '''Empty Trainer'''

    @property
    def parameters(self):
        """
        Returns the trainer parameters of the trainer.
        """
        raise FAPSTrainerException("The parameters property was not implemented.")

    @property
    def get_max_steps(self):
        """
        Returns the maximum number of steps. Is used to know when the trainer should be stopped.
        :return: The maximum number of steps of the trainer
        """
        raise FAPSTrainerException("The get_max_steps property was not implemented.")

    @property
    def get_step(self):
        """
        Returns the number of steps the trainer has performed
        :return: the step count of the trainer
        """
        raise FAPSTrainerException("The get_step property was not implemented.")

    @property
    def get_last_reward(self):
        """
        Returns the last reward the trainer has had
        :return: the new last reward
        """
        raise FAPSTrainerException("The get_last_reward property was not implemented.")

    def initialize(self):
        """
        Initialize the trainer
        """
        raise FAPSTrainerException("The initialize method was not implemented.")

    def is_initialized(self):
        """
        check if the trainer is initialized
        """
        return self.initialized

    def clear(self):
        """
        Clear the trainer
        """
        raise FAPSTrainerException("The clear method was not implemented.")

    def load_model_and_restore(self, model_path):
        """
        Load and restore the model from a defined path.

        :param model_path: Random seed.
        """
        raise FAPSTrainerException("The load_model_and_restore method was not implemented.")

    def increment_step(self):
        """
        Increment the step count of the trainer
        """
        raise FAPSTrainerException("The increment_step method was not implemented.")

    def update_last_reward(self, reward):
        """
        Updates the last reward
        """
        raise FAPSTrainerException("The update_last_reward method was not implemented.")

    def take_action(self, brain_info):
        """
        Decides actions given state/observation information, and takes them in environment.
        :param brain_info: The BrainInfo from environment.
        :return: the action array and an object to be passed to add experiences
        """
        raise FAPSTrainerException("The take_action method was not implemented.")

    def add_experiences(self, curr_info, action_vector, next_info):
        """
        Adds experiences to each agent's experience history.
        :param action_vector: Current executed action
        :param curr_info: Current AllBrainInfo.
        :param next_info: Next AllBrainInfo.
        """
        raise FAPSTrainerException("The add_experiences method was not implemented.")

    def process_experiences(self, current_info, action_vector, next_info):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param current_info: Current BrainInfo.
        :param action_vector: Current executed action
        :param next_info: Next corresponding BrainInfo.
        """
        raise FAPSTrainerException("The process_experiences method was not implemented.")

    def end_episode(self):
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets.
        """
        raise FAPSTrainerException("The end_episode method was not implemented.")

    def is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to wether or not update_model() can be run
        """
        raise FAPSTrainerException("The is_ready_update method was not implemented.")

    def update_model(self):
        """
        Uses training_buffer to update model. Run back propagation.
        """
        raise FAPSTrainerException("The update_model method was not implemented.")

    def save_model(self, model_path):
        """
        Save the model architecture to i.e. Tensorboard.
        :param model_path: The path where the model will be saved.
        """
        raise FAPSTrainerException("The save_model method was not implemented.")

    def write_summary(self):
        """
        Saves training statistics to i.e. Tensorboard.
        """
        raise FAPSTrainerException("The write_summary method was not implemented.")

    def write_tensor_board_text(self, key, input_dict):
        """
        Saves text to Tensorboard.
        Note: Only works on tensorflow r1.2 or above.
        :param key: The name of the text.
        :param input_dict: A dictionary that will be displayed in a table on Tensorboard.
        """
        # try:
        #     s_op = tf.summary.text(key,
        #                            tf.convert_to_tensor(([[str(x), str(input_dict[x])] for x in input_dict]))
        #                            )
        #     s = self.sess.run(s_op)
        #     self.summary_writer.add_summary(s, self.get_step)
        # except:
        #     logger.info("Cannot write text summary for Tensorboard. Tensorflow version must be r1.2 or above.")
        raise FAPSTrainerException("The write_tensor_board_text method was not implemented.")


pass
