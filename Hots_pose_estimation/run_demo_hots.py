import os
import sys

# Add to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FoundationPose')))

from utils.config import DemoConfig
from utils.demo import DemoRunner

if __name__ == "__main__":
    config = DemoConfig()
    runner = DemoRunner(config)
    runner.run()