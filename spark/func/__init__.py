from os.path import abspath
from configparser import ConfigParser
from pyspark.sql import SparkSession

# Load Configuration File
config = ConfigParser()
config.read(abspath('config.ini'))

# Initialize Spark
spark = SparkSession.builder.appName('ui').config("spark.jars", "/home/ubuntu/sparkclass/jar/postgresql-42.2.9.jar").getOrCreate()