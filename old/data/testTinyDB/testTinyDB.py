from tinydb import TinyDB, Query



db = TinyDB('test.json')

db.insert({'name': '1545a', 'candidates': ['4512b', '562a', '7845']})
