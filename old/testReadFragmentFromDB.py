import Fragment
from tinydb import TinyDB, Query

db = TinyDB('testFrag.json')

for f in db:
    a = Fragment.Fragment()
    a.loadFromTinyDB(f)
    print a.toString()
