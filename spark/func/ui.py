import os
from pyspark.sql.functions import column
from pyspark.sql.functions import lit
from .utils import ReadCSVFile
from .utils import ReadJsonFile

class UITable(object):
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
        self.csv = orig.select(column('UI Number').alias('uiid'),
                               column('App Package Name').alias('appname'),
                               column('Interaction Trace Number').alias('trace'),
                               column('UI Number in Trace').alias('uicount'))
        self.num = orig.count()
#         self.img_path = csv_task.path()
    def load(self):
        # load DataFrame
        result = self.csv
        return result
#     def add(self, folder):
#         num = self.num
# #         img_path = self.img_path + folder
#         img_path = folder
#         df = self.csv
# #         imglink = df
# #         imglink = df.select('*',(img_path + df.uiid + '.png').alias('img'))
#         imglink = df.select('*',(df.uiid+'.png').alias('img'))
#         imglink.show()
#         return imglink
