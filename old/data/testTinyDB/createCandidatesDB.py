from tinydb import TinyDB, Query

fragments = []

#add all the fragments from the test set : first argument is the fragment name, all the others are it's neighbors' names
addEntry('a', '78b', '45a', '95r')

db = TinyDB('neighborhood.json')
db.purge_tables()

for entry in fragments:
    db.insert(entry)

def addEntry(fragment, *neighbors):
    global fragments
    fragments.append({'name' : fragment, 'neighbors' : neighbors})
