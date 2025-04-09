import os
import sys

# Add to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FoundationPose')))

from utils.config import DemoConfig, LinemodConfig, PIPELINE_MODE
from utils.demo import DemoRunner
from utils.linemod import LinemodRunner

if __name__ == "__main__":
    # Choose the configuration based on the desired mode
    if PIPELINE_MODE == "demo":
        config = DemoConfig()
        runner = DemoRunner(config)
    else:
        config = LinemodConfig()
        runner = LinemodRunner(config)

    runner.run()