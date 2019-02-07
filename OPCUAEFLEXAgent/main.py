import sys
import time
import logging
from OPCUAEFLEXAgent.EFLEXAgent import EFLEXAgent


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

    # now setup our server
    server = EFLEXAgent(port=4840)

    # starting!
    server.start()

    try:
        if interactive:
            embed()
        else:
            while True:
                time.sleep(0.5)

    except IOError:
        pass
    finally:
        server.stop()
        print("done")

