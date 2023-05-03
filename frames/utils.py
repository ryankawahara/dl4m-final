from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import keras
import cv2
import os
import json
import random


def load_data(data_path, labels_dict, single=False):
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
        print("no dir", data_path)
        return False
    image_files = []
    labels = []
    ids = []

    for root, dirs, files in os.walk(data_path):
        for file in files[:10]:
            if file.endswith(('.png',)):
                id = file.split("_")

                # id = [item.decode('utf-8') for item in id]
                id = "_".join(id[:-1])
                genre_label = labels_dict[id]
                if single:
                    genre_label = set_one_random(genre_label)

                labels.append(genre_label)

                file_path = os.path.join(data_path, file)
                image_files.append(file_path)

                ids.append(id)

    # print(len(image_files) == len(labels))
    # return image_files, labels
    return image_files, labels, ids


def data_generator(data_path, img_shape, augment, normalize, shuffle, single=False):
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
    with open('/content/drive/My Drive/dl4m_datasets/trailer_model_data/genres.json', 'r') as f:
        # Load the JSON data as a dictionary
        labels_dict = json.load(f)

    image_files, labels, _ = load_data(str(data_path), labels_dict, single)

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

def commonly_associated_genres():
    with open(genres_path, 'r') as f:
        # Load the JSON data as a dictionary
        labels_dict = json.load(f)

    image_files, tr_labels, _ = u.load_data(os.path.join(data_home, "train", "frames"), labels_dict, single=False)
    image_files, tst_labels, _ = u.load_data(os.path.join(data_home, "test", "frames"), labels_dict, single=False)
    image_files, val_labels, _ = u.load_data(os.path.join(data_home, "validation", "frames"), labels_dict, single=False)



    all_labels = tr_labels[9::10] + tst_labels[9::10] + val_labels[9::10]
    print(len(all_labels))
    print(len(tst_labels[9::10]))
    print(len(val_labels[9::10]))
    save_path = os.path.join(data_home, "trailer_model_data", 'all_labels.npy')
    np.save(save_path, all_labels)

def create_dataset(data_path, batch_size, img_shape, augment=False, normalize=True, shuffle=True, single=False):
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
        args=[str(data_path), img_shape, augment, normalize, shuffle, single],
        output_signature=(
            tf.TensorSpec(shape=output_size, dtype=tf.float32),
            tf.TensorSpec(shape=(10,), dtype=tf.uint8)))

    # Add augmented images

    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset





def explore_data(train_ds, data_home, labels_dict,
                 class_names=['Action', 'Adventure', 'Comedy', 'Crime', 'Drama', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller'],
                 single=False
                 ):
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

    train_data_path = os.path.join(data_home, 'train', 'frames')
    image_files, labels, train_ids = load_data(train_data_path, labels_dict, single)
    training_ids = np.array(train_ids)
    # Save the array as an .npy file
    np.save('train_ids.npy', training_ids)


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




    validation_data_path = os.path.join(data_home, 'validation', 'frames')
    image_files, labels, v_ids = load_data(validation_data_path, labels_dict, single)
    validation_ids = np.array(v_ids)
    # Save the array as an .npy file
    np.save('validation_ids.npy', validation_ids)


    validation_class_counts = [0] * len(class_names)  # Initialize a list of counters, one for each genre

    for sublist in labels:
        for i, val in enumerate(sublist):
            if val == 1:
                validation_class_counts[i] += 1

    # train_class_counts = [[class_names[i] for i in range(len(label)) if label[i] == 1] for label in labels]

    ax[1].bar(range(len(class_names)), (validation_class_counts))
    ax[1].set_xticks(range(len(class_names)))
    ax[1].set_xticklabels(class_names, rotation=45)
    ax[1].set_title('Validation set')


    test_data_path = os.path.join(data_home, 'test', 'frames')
    image_files, labels, tst_ids = load_data(test_data_path, labels_dict, single)
    test_ids = np.array(tst_ids)
    # Save the array as an .npy file
    np.save('test_ids.npy', test_ids)



    test_class_counts = [0] * len(class_names)  # Initialize a list of counters, one for each genre

    for sublist in labels:
        for i, val in enumerate(sublist):
            if val == 1:
                test_class_counts[i] += 1

    # train_class_counts = [[class_names[i] for i in range(len(label)) if label[i] == 1] for label in labels]

    ax[2].bar(range(len(class_names)), (test_class_counts))
    ax[2].set_xticks(range(len(class_names)))
    ax[2].set_xticklabels(class_names, rotation=45)
    ax[2].set_title('Test set')

    print(test_class_counts, validation_class_counts)

    plt.show()



def convert_genres(genres):
    genre_list = []
    genre_names = ['Action', 'Adventure', 'Comedy', 'Crime', 'Drama', 'Horror', 'Mystery', 'Romance', 'Science Fiction', 'Thriller']

    for i in range(len(genres)):
        if genres[i] == 1:
            genre_list.append(genre_names[i])

    return genre_list

def get_features_and_labels(dataset, conv_base):
    """
    Extracts features from a pre-trained convolutional base and
    returns them along with their corresponding labels.

    Parameters
    ----------
    dataset : tf.data.Dataset
        The dataset containing the images and their corresponding labels.
    conv_base : keras.Model
        The pre-trained convolutional base used for feature extraction.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two numpy arrays - the concatenated features and labels respectively.

    Notes
    -----
    This function expects that the input images have already been preprocessed according to the requirements
    of the pre-trained convolutional base. Specifically, it uses the `preprocess_input` function from the
    VGG16 module to preprocess the images.

    """

    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed_images = keras.applications.vgg16.preprocess_input(images)
        features = conv_base.predict(preprocessed_images)
        all_features.append(features)

        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)


def set_one_random(arr):
    # Get the indices where the value is 1
    one_indices = [i for i, val in enumerate(arr) if val == 1]

    # Choose a random index from the one_indices list
    if one_indices:
        random_index = random.choice(one_indices)

        # Set all other indices to 0
        for i in range(len(arr)):
            if i != random_index:
                arr[i] = 0

    return arr


def plot_loss(history):
    """
    Plot the training and validation loss and accuracy.

    Parameters
    ----------
    history : keras.callbacks.History
        The history object returned by the `fit` method of a Keras model.

    Returns
    -------
    None
    """

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()
