# -*- coding: utf-8 -*-
from func.utils import Database
from func.model import Model
from pyspark.sql.functions import column
import pickle
import time
        
def prefilter(db, test):
    app_query = 'SELECT * FROM app WHERE (category = \'{}\')'.format(test['category'])
    ui_query = 'SELECT * FROM ui WHERE (appname IN (SELECT appname FROM app WHERE (category = \'{}\')))'.format(test['category'])
    app_df = db.query(app_query)
    ui_df = db.query(ui_query)
    return app_df, ui_df

def load_from_db(db, test_input, is_prefilter = True):
    # load from metadata db. You can choose the whether to do prefiltering.
    if is_prefilter:
        app_df, ui_df = prefilter(db, test_input)
    else:
        app_df = db.load(table='app')
        ui_df = db.load(table='ui')
    return app_df, ui_df

if __name__ == '__main__':
    db = Database()
    # load in a test sample
    file = open('localdata/test_ui', 'rb')
    test_ui = pickle.load(file)
    file.close()
    # prefiltering by category
    time_start=time.time()
    app_df, ui_df = load_from_db(db, test_ui, is_prefilter = True)
    ui_targets = ui_df.select(column('uiid'),column('appname')).collect()
    time_end=time.time()
    print('Read from sql time cost',time_end-time_start,'s')
    
    # use the ML/DL models
    time_start=time.time()
    model_demo = Model(test_ui, ui_targets)
    sim_df_pd = model_demo.cos_similarity()
    sim_app_df = ui_df.select(column('uiid'),column('appname')).where(ui_df.uiid.isin(sim_df_pd.uiid.iloc[0:6].tolist()))
    sim_app = sim_app_df.collect()
    result_df = app_df.where(app_df.appname.isin([row.appname for row in sim_app]))
    outputs = result_df.join(sim_app_df,"appname","left").collect()
    time_end=time.time()
    print('Model time cost',time_end-time_start,'s')
    
    file = open('localdata/result_demo', 'wb')
    pickle.dump(outputs, file)
    file.close()
    
        
    
    
    
    
