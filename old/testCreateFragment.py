import Fragment
from tinydb import TinyDB, Query

a = Fragment.Fragment()

print a.toString()

a.name = 'secondFrag'
a.IRR_file = 'test2.png'

db = TinyDB('testFrag.json')

print len(db)
a.saveToTinyDB(db)
print len(db)
