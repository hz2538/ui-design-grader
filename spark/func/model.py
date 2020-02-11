import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class Model(object):
    def __init__(self, user_input, ui_targets):
        self.input = user_input
        self.targets = ui_targets
        self.preload()
    
    def preload(self):
        # load from localdata file
        self.ui_vec = np.load('localdata/ui_vectors.npy')
        with open('localdata/ui_names.json') as f:
            data = json.load(f)
        self.ui_names = data['ui_names']
    
    def createdf(self):
        ui_vec = self.ui_vec
        ui_names = self.ui_names
        rows_list = []
        for i in range(len(ui_names)):
            row_dict={'uiid':ui_names[i].split('.')[0], 'vector':ui_vec[i]}
            rows_list.append(row_dict)
        df = pd.DataFrame(rows_list)
        return df 
    
    def cos_similarity(self):
        # the model of cosine similarity, return the indices of top10 similar UIs in pandas DataFrame.
        targets= self.targets
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
    