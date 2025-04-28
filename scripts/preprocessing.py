import os

# Get the directory where THIS script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get project root
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Flicker8k_Dataset")

