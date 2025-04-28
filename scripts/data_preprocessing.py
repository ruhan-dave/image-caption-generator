# Imports
import os
import glob
import random
import shutil

# Constants
# Set your dataset path
DATASET_PATH = "data/raw/Flicker8k_Dataset"
RANDOM_SEED = 100  # For reproducibility
TRAIN_RATIO = 0.7 #
VAL_RATIO = 0.2 #

# Get the directory where THIS script lives
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# # Go up one level to get project root
# PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Flicker8k_Dataset")


# --- Function Definitions ---

def create_split_dirs(base_path):
    """Creates train, val, and test directories within the base path."""
    # Define output directory paths
    train_dir = os.path.join(base_path, "train")
    val_dir = os.path.join(base_path, "val")
    test_dir = os.path.join(base_path, "test")

    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print(f"Created/Ensured directories exist: {train_dir}, {val_dir}, {test_dir}")
    return train_dir, val_dir, test_dir

def get_image_paths(dataset_path, pattern="*.jpg"):
    """Gets all image paths matching the pattern in the dataset path."""
    # Get all image paths (assuming .jpg by default)
    image_paths = glob.glob(os.path.join(dataset_path, pattern))
    print(f"Found {len(image_paths)} images.") #
    return image_paths

def split_data(image_paths, train_ratio, val_ratio, seed):
    """Shuffles and splits image paths into train, val, and test sets."""
    # Randomly shuffle the image list for random split
    random.seed(seed)
    random.shuffle(image_paths)

    # Calculate split sizes
    n_total = len(image_paths)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    # n_test is implicitly the remainder

    # Randomly assign images to each set
    train_images = image_paths[:n_train]
    val_images = image_paths[n_train : n_train + n_val]
    test_images = image_paths[n_train + n_val :]

    print(f"Training: {len(train_images)}") #
    print(f"Validation: {len(val_images)}") #
    print(f"Testing: {len(test_images)}") #
    return train_images, val_images, test_images

def copy_files(image_list, destination_dir):
    """Copies a list of image files to the destination directory."""
    print(f"Copying {len(image_list)} files to {destination_dir}...")
    for img_path in image_list: #
        try:
            # Copy file
            shutil.copy(img_path, os.path.join(destination_dir, os.path.basename(img_path)))
        except Exception as e:
            # Basic error handling
            print(f"Error copying {os.path.basename(img_path)}: {e}")
    print(f"Finished copying files to {destination_dir}.")


# --- Main Execution Logic ---

def main():
    """Main function to orchestrate the data splitting process."""
    # Create output directories
    train_dir, val_dir, test_dir = create_split_dirs(DATASET_PATH)

    # Get all image paths
    all_image_paths = get_image_paths(DATASET_PATH)

    if not all_image_paths:
        print("No images found. Exiting.")
        return

    # Split image paths into train, validation, and test sets
    train_images, val_images, test_images = split_data(
        all_image_paths, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED
    )

    # Copy files to their respective directories
    copy_files(train_images, train_dir)
    copy_files(val_images, val_dir)
    copy_files(test_images, test_dir)

    print("\nRandom splitting complete. Images are now in 'train', 'val', and 'test' subfolders.") #

# --- Script Entry Point ---

if __name__ == "__main__":
    main()