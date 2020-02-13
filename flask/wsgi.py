# -*- coding: utf-8 -*-

# Launch the server
from app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
