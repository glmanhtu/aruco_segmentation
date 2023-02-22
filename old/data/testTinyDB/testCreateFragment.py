import Fragment
from tinydb import TinyDB, Query

a = Fragment.Fragment()

print a.toString()

db = TinyDB('testFrag.json')

#print db.count(db.all)
#a.saveToTinyDB(db)
#print db.count(db.all)
