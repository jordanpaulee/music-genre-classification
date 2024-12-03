import os

data_dir = "data/raw/"
genres = os.listdir(data_dir)

def explore_dataset(data_dir):
    """
    Explore the dataset to verify the structure and count the files per genre.
    Args:
        data_dir (str): Path to the dataset directory.
    Returns:
        None
    """
    genres = os.listdir(data_dir)
    print("Dataset Overview:")
    for genre in genres:
        genre_path = os.path.join(data_dir, genre)
        if os.path.isdir(genre_path):
            num_files = len(os.listdir(genre_path))
            print(f"Genre: {genre}, Files: {num_files}")
        else:
            print(f"Warning: {genre_path} is not a directory.")

if __name__ == "__main__":
    DATA_DIR_ORG = "data/raw/genres_original"  # Path to raw audio dataset
    explore_dataset(DATA_DIR_ORG)

    print(f" ")

    DATA_DIR_IMG = "data/raw/images_original"  # Path to raw image dataset
    explore_dataset(DATA_DIR_IMG)
