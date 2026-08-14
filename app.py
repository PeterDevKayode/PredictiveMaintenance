import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pdm_pipeline", "src"))

from dashboard.app import *  # noqa: F401,F403
