# -*- coding: utf-8 -*-
# save the static tables to Postgres database.
from func.utils import *
from func.metadata import AppTable, UITable
from func.ui_semantic import SemTable
    
if __name__ == '__main__':
    app_path = 'app_details.csv'
    ui_path = 'ui_details.csv'
    db = Database()
    # save to database
    app_table = AppTable(app_path)
    ui_table = UITable(ui_path)
    js_task = SemTable()
    app_table.save(db)
    print("save app table success!")
    ui_table.save(db)
    print("save ui table success!")
    js_task.save(db)
    print("save semantic table success!")
    
        
    
    
    
    
