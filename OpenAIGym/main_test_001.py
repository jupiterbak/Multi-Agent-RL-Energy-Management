import sys
import time
import logging
import gym
import gym_eflex_agent


sys.path.insert(0, "..")

try:
    from IPython import embed
except ImportError:
    import code

    def embed():
        vars = globals()
        vars.update(locals())
        shell = code.InteractiveConsole(vars)
        shell.interact()

interactive = True

if __name__ == "__main__":
    # optional: setup logging
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger("opcua.address_space")
    logger.setLevel(logging.DEBUG)

    # test gym
    env = gym.make('eflex-agent-v0')
    observation = env.reset()
    for _ in range(5000):
        observation, reward, done, info = env.step(env.action_space.sample())
        print(info['info'])
        env.render('human')
    env.close()

    # try:
    #     if interactive:
    #         embed()
    #     else:
    #         while True:
    #             time.sleep(0.5)
    #
    # except IOError:
    #     pass
    # finally:
    #     print("done")

