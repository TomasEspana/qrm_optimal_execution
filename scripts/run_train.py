from my_project.config import load_config
from my_project.runner import RLRunner

if __name__ == "__main__":
    cfg = load_config("configs/default.yaml")
    runner = RLRunner(cfg)
    runner.run()