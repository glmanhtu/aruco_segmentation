#!/usr/bin/python3

import argparse
from tinydb import TinyDB, Query
from utils.fragment import Fragment
import shutil
import data.dataconfig as dataconfig

def listFunc(args):
    listDirectory(args.directory, args.valid, args.rejected, args.unclassified)

def validateFunc(args):
    updateFragments(args.directory, Fragment.PROCESS_STATE_VALID,
                    False,
                    args.allUnclassified,
                    args.allRejected,
                    args.name)
def rejectFunc(args):
    updateFragments(args.directory, Fragment.PROCESS_STATE_REJECTED,
                    args.allValid,
                    args.allUnclassified,
                    False,
                    args.name)

def restoreFunc(args):
    updateFragments(args.directory, Fragment.PROCESS_STATE_DEFAULT,
                    args.allValid,
                    False,
                    args.allRejected,
                    args.name)

def importFunc(args):
    importFragments(args.directory)
    

argParser = argparse.ArgumentParser()
argParser.add_argument('directory', type=str)
subParsers = argParser.add_subparsers()
subParsers.required = True
subParsers.dest = 'operation'

parser_list = subParsers.add_parser('list', help="lists fragments in this directory")
list_group = parser_list.add_mutually_exclusive_group()
parser_list.add_argument('-v', '--valid', action='store_true')
parser_list.add_argument('-r', '--rejected', action='store_true')
parser_list.add_argument('-u', '--unclassified', action='store_true')
parser_list.set_defaults(func=listFunc)

parser_validate = subParsers.add_parser('validate', help="mark the specified fragments as correctly processed and move them to the valid subdirectory")
parser_validate.add_argument('-aU', '--allUnclassified', action='store_true')
parser_validate.add_argument('-aR', '--allRejected', action='store_true')
parser_validate.add_argument('name', nargs='*')
parser_validate.set_defaults(func=validateFunc)

parser_reject = subParsers.add_parser('reject', help="mark the specified fragments as not correctly processed and move them to the rejected subdirectory")
parser_reject.add_argument('-aV', '--allValid', action='store_true')
parser_reject.add_argument('-aU', '--allUnclassified', action='store_true')
parser_reject.add_argument('name', nargs='*')
parser_reject.set_defaults(func=rejectFunc)

parser_restore = subParsers.add_parser('restore', help="mark the specified fragments as not classified and restore them to the root directory")
parser_restore.add_argument('-aV', '--allValid', action='store_true')
parser_restore.add_argument('-aR', '--allRejected', action='store_true')
parser_restore.add_argument('name', nargs='*')
parser_restore.set_defaults(func=restoreFunc)

parser_import = subParsers.add_parser('import', help="import all validated fragments in the directory to the main database")
parser_import.set_defaults(func=importFunc)


args = argParser.parse_args()
                      
def getDB(dir_name, db_name):
    db = TinyDB(dir_name+db_name)
    query = Query()
    return db, query
    
def listDirectory(dir_name, valid, rejected, unclassified):
    db, query = getDB(dir_name, "frags.json")
    res = []
    if(not valid and not rejected and not unclassified):
        valid = rejected = unclassified = True
        
    if(valid):
        res += db.search(query.processState == 'valid')
    if(rejected):
        res += db.search(query.processState == 'rejected')
    if(unclassified):
        res += db.search(query.processState == 'unclassified')
        
    for f in res:
        print("{} - {}".format(f['name'],
                               f['processState']))

def updateFragments(dir_name, state, allValid, allUnclassified, allRejected, names):
    db, query = getDB(dir_name, "frags.json")
    updt = {'processState': state}
    
    if(allValid):
        db.update(updt, query.processState == 'valid')
    if(allUnclassified):
        db.update(updt, query.processState == 'unclassified')
    if(allRejected):
        db.update(updt, query.processState == 'rejected')
    for n in names:
        db.update(updt, query.name == n)

def importFragments(dir_name):
    db,query = getDB(dir_name, "frags.json")
    mainDB, mainQ = getDB(dataconfig.DATA_PATH, dataconfig.FRAGMENT_TINYDB)
    print(mainDB.all())

    frags = db.search(query.processState == Fragment.PROCESS_STATE_VALID)

    for frag in frags:
        f = Fragment()
        f.loadFromTinyDB(frag)
        f.id = None
        f.saveToTinyDB(mainDB)

        shutil.rmtree(dataconfig.DATA_PATH+dataconfig.FRAGMENT_DIRECTORY+f.name, ignore_errors=True)
        shutil.copytree(dir_name+dataconfig.FRAGMENT_DIRECTORY+f.name,
                    dataconfig.DATA_PATH+dataconfig.FRAGMENT_DIRECTORY+f.name)       
            
args.func(args)



