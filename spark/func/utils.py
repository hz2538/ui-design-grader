from . import spark
from . import config

class ReadCSVFile(object):
    def __init__(self,name):
        self.name = name
        bucket = config.get('AWS', 'bucket')
        self.path = 's3a://{}/rico/{}'.format(bucket, self.name)
        self.imgpath = 's3a://{}/rico/semantic_annotations/'.format(bucket)
    def load(self):
        csv = spark.read.csv(self.path, header=True)
        return csv
    def img_path(self):
        return self.imgpath
    
class ReadJsonFile(object):
    def __init__(self,name):
        self.name = name
        bucket = config.get('AWS', 'bucket')
        self.path = 's3a://{}/rico/{}'.format(bucket, self.name)
    def load(self):
        js = spark.read.json(self.path, multiLine=True)
        return js

class Database(object):
    def __init__(self):
        self.username = config.get('PostgreSQL', 'username')
        self.password = config.get('PostgreSQL', 'password')

        instance = config.get('PostgreSQL', 'instance')
        database = config.get('PostgreSQL', 'database')
        self.url = 'jdbc:postgresql://{}:5432/{}'.format(instance, database)

    def save(self, data, table, mode='append'):
        data.write.format('jdbc') \
        .option("url", self.url) \
        .option("dbtable",table) \
        .option("user", self.username) \
        .option("password",self.password) \
        .option("driver", "org.postgresql.Driver") \
        .mode(mode).save()

    def load(self, table):
        return spark.read.format('jdbc') \
        .option("url", self.url) \
        .option("dbtable",table) \
        .option("user", self.username) \
        .option("password",self.password) \
        .option("driver", "org.postgresql.Driver") \
        .load()

    def query(self, query):
        return self.load('({}) AS frame'.format(query))