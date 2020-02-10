import os
from pyspark.sql.functions import column
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from .utils import ReadCSVFile
from .utils import ReadJsonFile
from .utils import Database
import pickle

class UITable(object):
    def __init__(self, path):
        self.path = path
        self.orig = None
        self.csv = None
        self.preprocess()
        
    def preprocess(self):
        # read from csv file, load as DataFrame
        csv_task = ReadCSVFile(self.path)
        imgfolder = csv_task.img_path()
        file = open('localdata/train_list', 'rb')
        train_list = pickle.load(file)
        train_id_list = [s.split('/')[1].split('.')[0] for s in train_list]
        file.close()
        orig = csv_task.load()
        self.orig = orig
        # rename columns
        csv_df = orig.select(column('UI Number').alias('uiid'),
                               column('App Package Name').alias('appname'),
                               column('Interaction Trace Number').alias('trace'),
                               column('UI Number in Trace').alias('uicount')).where(column("uiid").isin(train_id_list))
        writepath_udf = udf(lambda uiid: imgfolder + uiid + '.png', StringType())
        self.csv=csv_df.withColumn("path", writepath_udf(column("uiid")))
        self.num = self.csv.count()
    def load(self):
        # load DataFrame
        result = self.csv
        return result
    def save(self, db):
        ui_details = self.load()
        db.save(ui_details, table='ui')

class AppTable(object):
    '''
    The class is based on App DataFrame loaded from CSV file.
    '''
    def __init__(self, path):
        self.path = path
        self.orig = None
        self.csv = None
        self.preprocess()
    def preprocess(self):
        # read from csv file, load as DataFrame
        csv_task = ReadCSVFile(self.path)
        orig = csv_task.load()
        self.orig = orig
        # rename columns
        self.csv = orig.select(column('App Package Name').alias('appname'),
                               column('Play Store Name').alias('name'),
                               column('Category').alias('category'),
                               column('Average Rating').alias('rating'),
                               column('Number of Ratings').alias('ratenum'),
                               column('Number of Downloads').alias('dlnum'),
                               column('Date Updated').alias('update'),   
                               column('Icon URL').alias('url'))
    def load(self):
        # load DataFrame
        result = self.csv
        return result
    
    def save(self, db):
        app_details = self.load()
        db.save(app_details, table='app')
        