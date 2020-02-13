import os
import random
import numpy as np
import tensorflow as tf

def read_image_list_file(path, is_test):
    """
    Read all training/test images' path in the main image folder.
    (Input) path: str               the path of the main image folder.
    (Input) is_test: boolean        the flag of phase. True for test phase, False for training phase 
    (Output) list_image: list       the list that contains all selected images' path.
    """
    files = next(os.walk(path))[2]
    file_num = len(files)
    # stablize the random state, to make sure that the split is always the same.
    random.seed(999)
    imgdir_array = np.array(os.listdir(path))
    file_num = len(imgdir_array)
    id_list = random.sample(range(file_num),file_num)
    # current split percentage: 98% training, 2% test
    train_id_list = id_list[:int(0.98* file_num)]
    test_id_list = id_list[int(0.98* file_num):]
    train_array = imgdir_array[train_id_list]
    test_array = imgdir_array[test_id_list]
    list_image = []
    if is_test == False:
        for file_name in train_array:
            list_image.append(os.path.join(path,file_name))
    else:
        for file_name in test_array:
            list_image.append(os.path.join(path,file_name))

    return list_image

def preprocess_image(image):
    # preprocess the image. Read in, resize, and normalize.
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 128])
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_image(path):
    # read the image as tensor from the path, and preprocess it.
    image = tf.io.read_file(path)
    return preprocess_image(image)



