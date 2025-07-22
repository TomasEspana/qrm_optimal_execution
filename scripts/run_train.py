import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from qrm_rl.configs.config import load_config
from qrm_rl.runner import RLRunner


if __name__ == "__main__":

    config = load_config()
    runner = RLRunner(config)
    runner.run()
