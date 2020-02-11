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
    def __init__(self, user_input):
        
        self.input = user_input
        self.model = None
    
    def feature_load(self, vec_path, names_path):
        # latent space feature that loaded from localdata file, which is pretrained.
        self.ui_vec = np.load(vec_path)
        with open(names_path) as f:
            data = json.load(f)
        self.ui_names = data['ui_names']
    
    def model_load(self, path):
        # load model files
#         try:
#             self.model = preloading(path)
#         except:
#             print("You have not yet setted up a model.")
        self.model = preloading(path,self.input)
    
    def createdf(self):
        rows_list = []
        for i in range(len(self.ui_names)):
            row_dict={'uiid':self.ui_names[i].split('.')[0], 'vector':self.ui_vec[i]}
            rows_list.append(row_dict)
        df = pd.DataFrame(rows_list)
        return df 
    
    # local model
    def cos_similarity(self, targets):
        # the benchmark model: cosine similarity, which returns the indices of top6 similar UIs in pandas DataFrame.
        try:
            vec_path = 'localdata/ui_vectors.npy'
            names_path = 'localdata/ui_names.json'
            self.feature_load(vec_path, names_path)
            df_pd = self.createdf()
            sim_list = []
            for i,target in enumerate(targets):
                try:
                    tar_vec = df_pd[df_pd['uiid']==target.uiid].vector.to_numpy()[0]
                    sim = cosine_similarity((self.input['vector'],tar_vec))[0][1]
                    row_dict={'uiid':target.uiid, 'similarity': sim}
                    sim_list.append(row_dict)
                except:
                    continue
            sim_df_pd = pd.DataFrame(sim_list)
            sim_df_pd.sort_values('similarity',inplace=True,ascending=False) 
            return sim_df_pd
        except:
            err = "Model failed. Please check your input!"
            print(err)
            return err
            
        
    
    # model on GPU
    def VAE(self,image):
        this_path = os.path.dirname(os.path.abspath(__file__))
        self.input = image
        vec_path = '{}/tf2vae/ui_vectors.npy'.format(this_path)
        names_path = '{}/tf2vae/ui_names.json'.format(this_path)
        self.feature_load(vec_path, names_path)
        img_in, img_vec = test(self.model, image)
        tree = BallTree(self.ui_vec)
        # a brute-force method to walk on latent space, can be optimized later
        dist, idx = tree.query(img_vec, k=200)
        id_candidates = idx[0]
#         dest_vec = self.ui_vec[id_candidates[199]]
#         gen_vec = img_vec + 0.9 * (dest_vec - img_vec)
        for candidate in id_candidates:
            dest_vec = self.ui_vec[candidate]
            gen_vec = img_vec + 0.9 * (dest_vec - img_vec)
#             img_gen = generate(self.model,gen_vec)
            try:
                gen_vecs = tf.concat([gen_vecs, gen_vec], 0)
            except:
                gen_vecs = gen_vec
        img_gen = generate(self.model,gen_vecs) 
        return img_in, img_gen
        
    def NN_similar(self, image):
        vec = generated_encode(self.model, image)
        tree = BallTree(self.ui_vec)
        _, idx = tree.query(vec, k=10)
        id_candidates = idx[0]
        results = [self.ui_names[i] for i in id_candidates]
        return results
        
        
        
    