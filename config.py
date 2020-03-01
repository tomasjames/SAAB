#!/usr/bin/python
import psycopg2
from configparser import ConfigParser
 
 
def config_file(db_init_filename='database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(db_init_filename)
 
    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, db_init_filename))
 
    return db

'''
def connect(params=config_file(), close=False):
    """ Connect to the PostgreSQL database server """
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
      
        # create a cursor
        cur = conn.cursor()
        
   # execute a statement
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')
 
        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)
       
       # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn:
            conn.close()
            print('Database connection closed.')
'''


def create_tables(db_init_filename):
    """ create tables in the PostgreSQL database"""
    commands = (
        """
        CREATE TABLE chain_storage (
            chain_id INTEGER PRIMARY KEY,
            walker REAL NOT NULL,
            chain REAL NOT NULL,
            N_SIO REAL NOT NULL,
            N_SO REAL NOT NULL
        )
        """,
    )
    conn = None
    try:
        # read the connection parameters
        params = config_file(db_init_filename)
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        # create table one by one
        for command in commands:
            cur.execute(command)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn:
            conn.close()