# -*- coding: utf-8 -*-
from func.utils import *
from func.app import AppTable
from func.ui import UITable

import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.sql.functions import column

def savetosql(app_path, ui_path, db):
    app_process = AppTable(app_path)
    app_details = app_process.load()
    ui_process = UITable(ui_path)
    ui_details = ui_process.load()
    db.save(app_details, table='app')
    db.save(ui_details, table='ui')
    
def createdf(ui_names, ui_vec):
    rows_list = []
    for i in range(len(ui_names)):
        row_dict={'uiid':ui_names[i].split('.')[0], 'vector':ui_vec[i]}
        rows_list.append(row_dict)
    df = pd.DataFrame(rows_list)
    return df 
        
def prefilter(ui_details, app_details, vec):
    test_app = ui_details.select(column('appname')).filter(ui_details.uiid == vec['uiid']).collect()
    app_cat = app_details.select(column('category')).filter(app_details.appname == test_app[0].appname).collect()
    app_targets = app_details.select(column('appname')).filter(app_details.category == app_cat[0].category)
    ui_targets = ui_details.join(app_targets, 'appname', 'leftsemi').select(column('uiid')).collect()
    return ui_targets

def similarity(df_pd, targets):
    sim_list = []
    for i,target in enumerate(targets):
        try:
            tar_vec = df_pd[df_pd['uiid']==target.uiid].vector.to_numpy()[0]
            sim = cosine_similarity((test_ui['vector'],tar_vec))[0][1]
            row_dict={'uiid':target.uiid, 'similarity': sim}
            sim_list.append(row_dict)
        except:
            continue
    df = pd.DataFrame(sim_list)
    df.sort_values('similarity',inplace=True,ascending=False) 
    return df

if __name__ == '__main__':
    app_path = 'app_details.csv'
    ui_path = 'ui_details.csv'
    json_path = 'unique_uis/0.json'
    img_folder = 'unique_uis'
#     json_task = ReadJsonFile (jsonname)
#     js = json_task.load()
    db = Database()
    # save metadata for the first time, then comment it
#     savetosql(app_path, ui_path, db)

    app_details = db.load(table='app')
    ui_details = db.load(table='ui')
    
    ui_vec = np.load('localdata/ui_vectors.npy')
    with open('localdata/ui_names.json') as f:
        data = json.load(f)
    ui_names = data['ui_names']
    # set a input vector
    test_ui= {'uiid': ui_names[0].split('.')[0], 'vector': ui_vec[0]}
    # prefiltering by category
    ui_targets = prefilter(ui_details, app_details, test_ui)
    # create panda dataframe
    df_pd = createdf(ui_names, ui_vec)
    # calculate similarity
    sim_df_pd = similarity(df_pd, ui_targets)
    outputs = []
    for i in range(10):
        sim_app = ui_details.select(column('appname')).filter(ui_details.uiid == sim_df_pd.loc[i].uiid).collect()
        output = app_details.filter(app_details.appname == sim_app[0].appname).collect()
        outputs.append(output)
    print(outputs)
        
    
    
    
    
