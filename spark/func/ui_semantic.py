from . import spark
from pyspark.sql.functions import column, lit
from pyspark.sql import Row
from .utils import ReadCSVFile
from .utils import ReadJsonFile
import pickle

class SemTable(object):
    '''
    The class is based on semantic segmantation json file.
    '''
    def __init__(self):
        file = open('localdata/train_list', 'rb')
        train_list = pickle.load(file)
        self.js_list = [(s.split('.')[0] + '.json') for s in train_list]
        
    def iterdict(self, d, elements):
        # this function iteratively extract elements from the nested json.
        for k,v in d.items():        
            if isinstance(v, dict):
                self.iterdict(v,elements)
            elif isinstance(v,list):
                for v_child in v:
                    if isinstance(v_child,dict):
                        self.iterdict(v_child,elements)
                        v_in = {'bounds':v_child['bounds']}
                        v_in.update({'componentLabel':v_child['componentLabel']})
                        elements.append(v_in)
    def getitem(self, file_path):
        # read from json file, load as DataFrame
        js_task = ReadJsonFile(file_path)
        uiid = file_path.split('/')[1].split('.')[0]
        js_df = js_task.load()
        new_rdd = js_df.rdd.map(lambda row: row.asDict(True))
        d = new_rdd.collect()[0]
        elements = []
        self.iterdict(d,elements)
        # deal with empty layouts
        try:
            element_df = spark.createDataFrame(Row(**x) for x in elements)
            clean_df=element_df.select(column('bounds'),column('componentLabel').alias('componentlabel'))
            self.df = clean_df.withColumn('uiid', lit(uiid))
        except:
            self.df = None
        return self.df
    def loadone(self):
        return self.df
    def save(self, db):
        for file_path in self.js_list:
            js_list = self.getitem(file_path)
            if js_list != None:
                db.save(js_list, table='semantic')