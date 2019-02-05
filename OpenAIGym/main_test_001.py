import sys
import time
import logging
import gym

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
    env = gym.make('FrozenLake-v0')
    env.reset()
    for _ in range(5000):
        env.step(env.action_space.sample())
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

