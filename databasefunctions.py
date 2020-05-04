#!/usr/bin/python
import psycopg2
from psycopg2 import pool
from configparser import ConfigParser


def dbpool(db_params):
    """
    A function that create a simple connection pool to a 
    Postgres database.

    Arguments:
        db_params {dict} -- A dictionary containing the 
            username, password, host name and database to connect to.

    Returns: 
        db_pool {object} --  a psycopg2 object containing the 
            connection pool.
    """

    db_pool = psycopg2.pool.SimpleConnectionPool(1, 8, 
                user=db_params["user"],
                password=db_params["password"],
                host=db_params["host"],
                database=db_params["database"])
    
    if(db_pool):
        print("Connection pool created successfully")

    return db_pool


def create_table(db_pool, commands):
    """Create tables in the PostgreSQL database"""
    conn = None
    try:
        # connect to the PostgreSQL server
        # Use getconn() to Get Connection from connection pool
        conn = db_pool.getconn()
        # conn = psycopg2.connect(**db_params)
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
            # Release the connection object back to the pool
            db_pool.putconn(conn)

	
def insert_chain_data(db_pool, table, chain):
    """ insert multiple vendors into the vendors table  """
    sql = "INSERT INTO {0} (id, vs, initial_dens) VALUES (DEFAULT, {1}, {2});".format(table, chain[0], chain[1])
    conn = None
    try:
        # connect to the PostgreSQL server
        # Use getconn() to Get Connection from connection pool
        conn = db_pool.getconn()
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        cur.execute(sql, table)
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        # cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            # Release the connection object back to the pool
            db_pool.putconn(conn)


def insert_data(db_pool, table, data):
    """ insert multiple vendors into the vendors table  """
    sql = """INSERT INTO {0} (id, species, transitions, vs, initial_n, resolved_T, resolved_n, N, radex_flux, source_flux, source_flux_error, chi_squared) VALUES ( DEFAULT, ARRAY [ '{1}', '{2}' ], ARRAY [ '{3}', '{4}' ], {5}, {6}, {7}, {8}, ARRAY [ {9}, {10} ], ARRAY [ {11}, {12} ], ARRAY [ {13}, {14} ], ARRAY [ {15}, {16} ], {17});""".format(
        table, str(data[0][0]), str(data[0][1]), str(data[1][0]), str(data[1][1]), data[2], data[3], data[4], data[5], data[6][0], data[6][1], data[7][0], data[7][1], data[8][0], data[8][1], data[9][0], data[9][1], data[10])
    conn = None
    try:
        # connect to the PostgreSQL server
        # Use getconn() to Get Connection from connection pool
        conn = db_pool.getconn()
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        cur.execute(sql, table)
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        # cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            # Release the connection object back to the pool
            db_pool.putconn(conn)


def get_chains(db_pool, table, column_names):
    """ query chains from the chain_storage table """
    conn = None
    try:
        # connect to the PostgreSQL server
        # Use getconn() to Get Connection from connection pool
        conn = db_pool.getconn()
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
        # cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print("ERROR THROWN")
        print(error)
        rows = None
    finally:
        if conn is not None:
            conn.close()
            # Release the connection object back to the pool
            db_pool.putconn(conn)
    return rows


def get_bestfit(db_pool, table, column_names):
    """ query chains from the chain_storage table """
    conn = None
    try:
        # connect to the PostgreSQL server
        # Use getconn() to Get Connection from connection pool
        conn = db_pool.getconn()
        cur = conn.cursor()
        cur.execute(
            "SELECT {0}, {1}, {2}, {3} FROM {4};".format(
                column_names[0],
                column_names[1],
                column_names[2],
                column_names[3],
                table
            )
        )
        rows = cur.fetchall()
        # rows = [r[0] for r in cur.fetchall()]
        print("The number of entries: ", cur.rowcount)
        # cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print("ERROR THROWN")
        print(error)
        rows = None
    finally:
        if conn is not None:
            conn.close()
            # Release the connection object back to the pool
            db_pool.putconn(conn)
    return rows


def does_table_exist(db_pool, table):
    conn = None
    try:
        # connect to the PostgreSQL server
        # Use getconn() to Get Connection from connection pool
        conn = db_pool.getconn()
        cur = conn.cursor()
        cur.execute('SELECT 1 from {0};'.format(table))
        ver = bool(cur.fetchone())
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        ver = False
    finally:
        if conn is not None:
            conn.close()
            # Release the connection object back to the pool
            db_pool.putconn(conn)
    return ver


def drop_table(db_pool, table):
    conn = None
    try:
        # connect to the PostgreSQL server
        # Use getconn() to Get Connection from connection pool
        conn = db_pool.getconn()
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
            # Release the connection object back to the pool
            db_pool.putconn(conn)
