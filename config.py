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


def create_table(db_init_filename, commands):
    """Create tables in the PostgreSQL database"""
    conn = None
    try:
        # read the connection parameters
        params = config_file()
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

	
def insert_chain_data(table, chain):
    """ insert multiple vendors into the vendors table  """
    sql = "INSERT INTO {0}(id, walker, chain, N_SIO, N_SO) VALUES (DEFAULT, {1}, {2}, {3}, {4});".format(table, chain[0], chain[1], chain[2], chain[3])
    conn = None
    try:
        # read database configuration
        params = config_file()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        cur.execute(sql, chain)
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()