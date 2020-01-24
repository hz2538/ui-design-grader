from pyspark.sql.functions import column
from pyspark.sql.functions import concat
from .utils import ReadCSVFile
from .utils import ReadJsonFile

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