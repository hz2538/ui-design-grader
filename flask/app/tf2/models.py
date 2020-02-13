import json
import os
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import BallTree
import sys 
this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_path)
from tf2vae.VariationalAutoEncoder import preloading, test, generate, generated_encode
class Model(object):
    """
    Model class integrates of all pre-trained deep learning models, and provides interfaces of using the pre-trained models.
    """
    def __init__(self, user_input):
        """
        Initialization.
        (Input) user_input: str   path to the user's input semantic annotation image. 
        """
        self.input = user_input
        self.model = None
    
    @property
    def feature_load(self, vec_path, names_path):
        # latent space feature that loaded from localdata file, which is pretrained.
        self.ui_vec = np.load(vec_path)
        with open(names_path) as f:
            data = json.load(f)
        self.ui_names = data['ui_names']
    
    @property
    def model_load(self, path):
        # load model files
        try:
            self.model = preloading(path)
        except:
            print("You have not yet set up a model.")
        
    # model on GPU
    def VAE(self,image):
        """
        Interface of Variational AutoEncoder model.
        (Input) image: str          path to the user's latest input semantic annotation image. This is required for updating the test image.
        (Output) img_in: Tensor     the Tensor format of the input image.
        (Output) img_gen: Tensor    the generated image by model.
        """
        this_path = os.path.dirname(os.path.abspath(__file__))
        self.input = image
        vec_path = '{}/tf2vae/ui_vectors.npy'.format(this_path)
        names_path = '{}/tf2vae/ui_names.json'.format(this_path)
        self.feature_load(vec_path, names_path)
        img_in, img_vec = test(self.model, image)
        tree = BallTree(self.ui_vec)
        # following is a brute-force method to walk on latent space, can be optimized later
        # 0.9 is set as a safety coordinate (0,1) to decide how "brave" to behave in walking to the target.
        # the smaller one means tend to be braver.
        dist, idx = tree.query(img_vec, k=200)
        id_candidates = idx[0]
        for candidate in id_candidates:
            dest_vec = self.ui_vec[candidate]
            gen_vec = img_vec + 0.9 * (dest_vec - img_vec)
            try:
                gen_vecs = tf.concat([gen_vecs, gen_vec], 0)
            except:
                gen_vecs = gen_vec
        img_gen = generate(self.model,gen_vecs) 
        return img_in, img_gen

    
    def AE(self,image):
        # to be updated
        return None
    def VAEGAN(self,image):
        # to be updated
        return None
    
    def NN_similar(self, image):
        """
        Similarity calculation using BallTree Nearest Neighbors algorithm.
        (Input) image: ndarray      The numpy ndarray generated image after computer vision correction.
        (Output) results: list      A list of candidates' uiid.    
        """
        vec = generated_encode(self.model, image)
        tree = BallTree(self.ui_vec)
        # indices of 10 closest neighbors as candidates
        _, idx = tree.query(vec, k=10)
        id_candidates = idx[0]
        results = [self.ui_names[i] for i in id_candidates]
        return results
        
        
        
    