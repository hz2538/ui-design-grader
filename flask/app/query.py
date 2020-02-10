# -*- coding: utf-8 -*-
from . import connection
    
class Validate(object):
    """
    Validate the similarity results

    """
    def __init__(self, candidates):
        """Initialization

        """
        self._cursor = connection.cursor()
        self.uiid_list = [c.split('.')[0] for c in candidates]
        self.uiids = tuple(self.uiid_list)
    def __del__(self):
        """Deletion

        Colse the cursor for database.

        """
        self._cursor.close()
        
    def fetch(self):
        """Fetch results 
        Validate if the candidates contains the added elements.

        """
        table = []
        for i in range(len(self.uiid_list)):
            sql = 'SELECT name, category, rating, dlnum FROM t1 WHERE (uiid = %(cands)s);'
            self._cursor.execute(sql,{'cands':self.uiids[i]})
            result=self._cursor.fetchall()
            table.append(result[0])
        uiid_list = self.uiid_list
        return table, uiid_list
        
        
