# -*- coding: utf-8 -*-

# --- flask ---
from flask import Flask

app = Flask(__name__)

# --- database ---
from os.path import abspath
from configparser import ConfigParser
from psycopg2 import connect

parser = ConfigParser()
parser.read(abspath('config.ini'))

config = {
    'host': parser.get('PostgreSQL', 'instance'),
    'database': parser.get('PostgreSQL', 'database'),
    'user': parser.get('PostgreSQL', 'username'),
    'password': parser.get('PostgreSQL', 'password'),
}
connection = connect(**config)

aws_id = parser.get('AWS', 'id')
aws_key = parser.get('AWS', 'key')
# --- interface ---
from . import interface