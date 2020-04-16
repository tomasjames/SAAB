#!/usr/bin/python
import psycopg2
from configparser import ConfigParser



def create_table(db_params, commands):
    """Create tables in the PostgreSQL database"""
    conn = None
    try:
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**db_params)
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

	
def insert_chain_data(db_params, table, chain):
    """ insert multiple vendors into the vendors table  """
    sql = "INSERT INTO {0}(id, vs, initial_dens) VALUES (DEFAULT, {1}, {2});".format(table, chain[0], chain[1])
    conn = None
    try:
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**db_params)
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        cur.execute(sql, table)
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def get_chains(db_params, table, column_names):
    """ query chains from the chain_storage table """
    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        cur.execute(
            "SELECT {0}, {1} FROM {2};".format(
                column_names[0], 
                column_names[1],  
                table
            )
        )
        rows = cur.fetchall()
        # rows = [r[0] for r in cur.fetchall()]
        print("The number of entries: ", cur.rowcount)
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print("ERROR THROWN")
        print(error)
        rows = None
    finally:
        if conn is not None:
            conn.close()
    return rows


def does_table_exist(db_params, table):
    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        cur.execute('SELECT 1 from {0};'.format(table))
        ver = bool(cur.fetchone())
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        ver = False
    finally:
        if conn is not None:
            conn.close()
    return ver


def drop_table(db_params, table):
    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()

        # Form the SQL statement - DROP TABLE
        cur.execute("DROP TABLE IF EXISTS {0};".format(table))
        conn.commit()
        print("Deletion successful")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
