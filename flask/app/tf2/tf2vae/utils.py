import os
import random
import numpy as np
import tensorflow as tf

def read_image_list_file(path, is_test):
    files = next(os.walk(path))[2]
    file_num = len(files)
    random.seed(999)
    imgdir_array = np.array(os.listdir(path))
    file_num = len(imgdir_array)
    id_list = random.sample(range(file_num),file_num)
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
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 128])
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)



