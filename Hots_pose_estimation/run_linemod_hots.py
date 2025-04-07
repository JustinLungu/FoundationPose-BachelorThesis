import os
import sys
# Add to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FoundationPose')))

from utils.config import LinemodConfig
from utils.linemod import LinemodRunner

if __name__ == "__main__":
    config = LinemodConfig()
    runner = LinemodRunner(config)
    runner.run()