# -*- coding: utf-8 -*-
from . import connection
    
class Validate(object):
    """
    Validate the similarity candidates. Find 4 most qualified similar candidates as results.

    """

    def __init__(self, candidates, labels):
        """Initialization

        """
        self._cursor = connection.cursor()
        self.uiid_cand_list = [c.split('.')[0] for c in candidates]
        self.uiids = tuple(self.uiid_cand_list)
        self.labels = tuple(labels)

    def __del__(self):
        """Deletion

        Close the cursor for database.

        """
        self._cursor.close()

    def fetch(self):
        """Fetch results from database.

        Fetch the validated results, and query corresponding app information for display. 

        """
        table = []
        uiid_list = []
        sql = 'SELECT uiid FROM semantic WHERE (uiid in %(cands)s AND componentLabel in %(comp)s)'
        self._cursor.execute(sql,{'cands':self.uiids, 'comp': self.labels})
        id_match=self._cursor.fetchall()
        sql = 'SELECT name, category, rating, dlnum, uiid FROM t1 WHERE (uiid in %(cands)s) ORDER BY rating DESC LIMIT 4;'
        self._cursor.execute(sql,{'cands':tuple(id_match)})
        table=self._cursor.fetchall()
        uiid_list = [table[i][-1] for i in range(len(table))]
        return table, uiid_list
        
        
