from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import keras
import cv2
import os
import json


def load_data(data_path, labels_dict):
    """
    Load image files and their corresponding labels from a given directory.

    Parameters
    ----------
    data_path : str
        Path to the directory containing the image files.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two numpy arrays. The first array contains the file paths of the image files, and the second array
        contains their corresponding labels (0 for cat and 1 for dog).
    """
    data_path = os.path.abspath(data_path)
    if os.path.exists(data_path) == False:
        print("no dir")
        return False
    image_files = []
    labels = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(('.png',)):
                id = file.split("_")

                # id = [item.decode('utf-8') for item in id]
                id = "_".join(id[:-1])
                genre_label = labels_dict[id]

                labels.append(genre_label)

                file_path = os.path.join(data_path, file)
                image_files.append(file_path)

    # print(len(image_files) == len(labels))
    return image_files, labels


def data_generator(data_path, img_shape, augment, normalize, shuffle):
    """
    A generator function that yields batches of preprocessed images and their corresponding labels from a directory of images.

    Parameters:
    -----------
    data_path : str
        Path to the directory containing images.
    img_shape : tuple
        Shape to which the images will be resized to.
    augment : bool
        Whether to perform data augmentation on the images or not.
    normalize : bool
        Whether to normalize the pixel values of the images or not. Defaults to True.
    shuffle : bool
        Whether to shuffle the data or not.

    Yields:
    -------
    tuple of numpy arrays
        A tuple containing the preprocessed image and its corresponding label.

    Example:
    --------
    data_gen = data_generator('/path/to/images', (224,224), True, True)
    images, labels = next(data_gen)
    """
    if isinstance(data_path, bytes):
        data_path = data_path.decode('utf-8')


    # Get list of image file names and their corresponding labels
    with open('genres.json', 'r') as f:
        # Load the JSON data as a dictionary
        labels_dict = json.load(f)

    image_files, labels = load_data(str(data_path), labels_dict)


    # Convert labels to numpy array
    labels = np.array(labels)



    # Shuffle images and labels
    if shuffle:
        idxs = np.random.permutation(len(labels))
        image_files = [image_files[i] for i in idxs]
        labels = labels[idxs]

    for idx in range(len(image_files)):

        # Load image and label
        label = labels[idx]
        file_path = image_files[idx]
        # NEED TO ADD THE FOLDER

        if isinstance(file_path, bytes):
            file_path = str(file_path.decode('utf-8'))

        path_components = file_path.split(os.path.sep)
        file = os.path.basename(file_path)

        file_folder_name = file.split("_")
        file_folder_name = "_".join(file_folder_name[:-1])

        path_components.insert(len(path_components) - 1, file_folder_name)
        new_path = os.path.sep.join(path_components)

        file_path = new_path

        img = cv2.imread(file_path)


        # Correct BGR to RGB color
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(e)
            continue

        crop = False

        # Resize image to expected size
        if crop == True:
            height, width = img.shape[:2]

            # Calculate the coordinates of the center of the image
            center_y = int(height / 2)
            center_x = int(width / 2)

            # Calculate the coordinates of the top-left corner of the crop
            crop_y = center_y - int(img_shape[0] / 2)
            crop_x = center_x - int(img_shape[1] / 2)

            # Crop the image
            cropped_img = img[crop_y:crop_y + img_shape[0], crop_x:crop_x + img_shape[1]]

            # Resize the cropped image to the desired shape
            img = cv2.resize(cropped_img, img_shape)
        else:
            img = cv2.resize(img, img_shape)

        # if augment:
        #     # Augment image
        #     img = augment_image(img)

        if normalize:
            # Normalize image to within [0,1]
            img = img / 255.


        yield img, label


def create_dataset(data_path, batch_size, img_shape, augment=False, normalize=True, shuffle=True):
    """
    Creates a TensorFlow dataset from image files in a directory.

    Parameters
    ----------
    data_path : str
        Path to directory containing the image files.
    batch_size : int
        Batch size for the returned dataset.
    img_shape : tuple
        Tuple of integers representing the desired image shape, e.g. (224, 224).
    augment : bool, optional
        Whether to apply image augmentation to the dataset. Default is False.
    normalize : bool, optional
        Whether to normalize the pixel values of the images to the range [0, 1]. Default is False.

    Returns
    -------
    dataset : tf.data.Dataset
        A TensorFlow dataset containing the images and their labels.
    """
    output_size = img_shape + (3,)

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        args=[str(data_path), img_shape, augment, normalize, shuffle],
        output_signature=(
            tf.TensorSpec(shape=output_size, dtype=tf.float32),
            tf.TensorSpec(shape=(10,), dtype=tf.uint8)))

    # Add augmented images
    if augment:
        dataset_aug = tf.data.Dataset.from_generator(
            data_generator,
            args=[data_path, img_shape, augment, normalize, shuffle],
            output_signature=(
                tf.TensorSpec(shape=output_size, dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.uint8)))
        dataset = dataset.concatenate(dataset_aug)

    dataset = dataset.batch(batch_size)

    return dataset


def create_dataset_from_file_paths(file_paths, batch_size, img_shape, augment=False, normalize=True, shuffle=True):
    """
    Creates a TensorFlow dataset from image files.

    Parameters
    ----------
    file_paths : list
        List of file paths containing the image files.
    batch_size : int
        Batch size for the returned dataset.
    img_shape : tuple
        Tuple of integers representing the desired image shape, e.g. (224, 224).
    augment : bool, optional
        Whether to apply image augmentation to the dataset. Default is False.
    normalize : bool, optional
        Whether to normalize the pixel values of the images to the range [0, 1]. Default is False.

    Returns
    -------
    dataset : tf.data.Dataset
        A TensorFlow dataset containing the images and their labels.
    """
    output_size = img_shape + (3,)
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(
        lambda x: tuple(tf.numpy_function(
            load_and_preprocess_image, [x, img_shape, augment, normalize], [tf.float32, tf.uint8])),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)

    return dataset



def explore_data(train_ds, data_home, class_names=['Action', 'Adventure', 'Comedy', 'Crime', 'Drama', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller']):
    """
    Plots the distribution of classes in the training, validation, and test sets, and displays a sample of images from the
    training set.

    Parameters
    ----------
    train_ds : tf.data.Dataset
        A dataset object for the training set.
    data_home : str
        The directory path to the dataset.
    class_names : List[str]
        A list of class names.

    Returns
    -------
    None
    """

    # Plot the distribution of classes in the training, validation, and test sets
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    # Plot the distribution of classes in the training set
    # image_files, labels = load_data(os.path.join(data_home, 'train'))
    with open('genres.json', 'r') as f:
        # Load the JSON data as a dictionary
        labels_dict = json.load(f)
    image_files, labels = load_data(data_home, labels_dict)

    # train_class_counts = [class_names[i] for i in range(len(labels)) if labels[i] == 1]
    # print(train_class_counts, labels)

    train_class_counts = [0] * len(class_names)  # Initialize a list of counters, one for each genre

    for sublist in labels:
        for i, val in enumerate(sublist):
            if val == 1:
                train_class_counts[i] += 1


    # train_class_counts = [[class_names[i] for i in range(len(label)) if label[i] == 1] for label in labels]

    ax[0].bar(range(len(class_names)), (train_class_counts))
    ax[0].set_xticks(range(len(class_names)))
    ax[0].set_xticklabels(class_names, rotation=45)
    ax[0].set_title('Training set')


    plt.show()



def convert_genres(genres):
    genre_list = []
    genre_names = ['Action', 'Adventure', 'Comedy', 'Crime', 'Drama', 'Horror', 'Mystery', 'Romance', 'Science Fiction', 'Thriller']

    for i in range(len(genres)):
        if genres[i] == 1:
            genre_list.append(genre_names[i])

    return genre_list
