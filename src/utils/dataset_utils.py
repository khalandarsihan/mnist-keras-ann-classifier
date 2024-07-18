import pickle
import json
import numpy as np
from pathlib import Path

def save_dataset(data, base_path, format='pickle'):
    """
    Saves the dataset in the specified format.

    :param data: tuple of (x_train, y_train, x_test, y_test)
    :param base_path: directory where the files will be saved
    :param format: 'pickle' or 'json'
    """
    x_train, y_train, x_test, y_test = data

    # Create directory if it does not exist
    Path(base_path).mkdir(parents=True, exist_ok=True)

    if format == 'pickle':
        with open(f"{base_path}/x_train.pkl", 'wb') as f:
            pickle.dump(x_train, f)
        with open(f"{base_path}/y_train.pkl", 'wb') as f:
            pickle.dump(y_train, f)
        with open(f"{base_path}/x_test.pkl", 'wb') as f:
            pickle.dump(x_test, f)
        with open(f"{base_path}/y_test.pkl", 'wb') as f:
            pickle.dump(y_test, f)
    elif format == 'json':
        with open(f"{base_path}/x_train.json", 'w') as f:
            json.dump(x_train.tolist(), f)
        with open(f"{base_path}/y_train.json", 'w') as f:
            json.dump(y_train.tolist(), f)
        with open(f"{base_path}/x_test.json", 'w') as f:
            json.dump(x_test.tolist(), f)
        with open(f"{base_path}/y_test.json", 'w') as f:
            json.dump(y_test.tolist(), f)
    else:
        raise ValueError("Format must be 'pickle' or 'json'")

    print(f"Dataset saved in {format} format at {base_path}")

def load_dataset(base_path, format='pickle'):
    """
    Loads the dataset in the specified format.

    :param base_path: directory where the files are saved
    :param format: 'pickle' or 'json'
    :return: tuple of (x_train, y_train, x_test, y_test)
    """
    if format == 'pickle':
        with open(f"{base_path}/x_train.pkl", 'rb') as f:
            x_train = pickle.load(f)
        with open(f"{base_path}/y_train.pkl", 'rb') as f:
            y_train = pickle.load(f)
        with open(f"{base_path}/x_test.pkl", 'rb') as f:
            x_test = pickle.load(f)
        with open(f"{base_path}/y_test.pkl", 'rb') as f:
            y_test = pickle.load(f)
    elif format == 'json':
        with open(f"{base_path}/x_train.json", 'r') as f:
            x_train = np.array(json.load(f))
        with open(f"{base_path}/y_train.json", 'r') as f:
            y_train = np.array(json.load(f))
        with open(f"{base_path}/x_test.json", 'r') as f:
            x_test = np.array(json.load(f))
        with open(f"{base_path}/y_test.json", 'r') as f:
            y_test = np.array(json.load(f))
    else:
        raise ValueError("Format must be 'pickle' or 'json'")

    print(f"Dataset loaded from {base_path}")
    return x_train, y_train, x_test, y_test


def check_missing_data(x_data, y_data):
    missing_count = 0
    for image, label in zip(x_data, y_data):
        if image is None or label is None:
            missing_count += 1
    return missing_count


import tensorflow as tf
def preprocess_mnist(x_data, y_data, img_height=28, img_width=28):
    # Normalize pixel values to be between 0 and 1
    x_data = x_data.astype('float32') / 255.0
    
    # Resize images if necessary
    if img_height != 28 or img_width != 28:
        x_data = tf.image.resize(x_data, [img_height, img_width])
    
    return x_data, y_data
