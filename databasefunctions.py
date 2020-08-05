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


def insert_radex_chain_data(db_pool, table, chain, column_names):
    """ insert multiple vendors into the vendors table  """
    sql = "INSERT INTO {0} (id, {1}) VALUES (DEFAULT, {2});".format(
        table, ", ".join(column_names), ', '.join(map(str, chain)))
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

	
def insert_shock_chain_data(db_pool, table, chain):
    """ insert multiple vendors into the vendors table  """
    sql = "INSERT INTO {0} (id, vs, dens, B_field, crir, isrf) VALUES (DEFAULT, {1}, {2}, {3}, {4}, {5});".format(table, chain[0], chain[1], chain[2], chain[3], chain[4])
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
    sql = """INSERT INTO {0} (id, species, transitions, temp, dens, column_density, radex_flux, source_flux, source_flux_error, chi_squared) VALUES ( DEFAULT, ARRAY {1}, ARRAY {2}, {3}, {4}, ARRAY {5}, ARRAY {6}, ARRAY {7}, ARRAY {8}, {9} );""".format(
        table, data["species"], data["transitions"], data["temp"], data["dens"], data["column_density"], data["rj_flux"], data["source_rj_flux"], data["source_rj_flux_error"], data["chi"])
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


def insert_shock_data(db_pool, table, data):
    """ insert multiple vendors into the vendors table  """
    sql = """INSERT INTO {0} (id, species, transitions, vs, dens, b_field, crir, isrf, column_density, resolved_T, resolved_n, radex_flux, source_flux, source_flux_error, chi_squared) VALUES ( DEFAULT, ARRAY {1}, ARRAY {2}, {3}, {4}, {5}, {6}, {7}, ARRAY {8}, {9}, {10}, ARRAY {11}, ARRAY {12}, ARRAY {13}, {14} );""".format(
        table, data["species"], data["transitions"], data["vs"], data["dens"], data["b_field"], data["crir"], data["isrf"], data["column_density"], data["resolved_T"], data["resolved_n"], data["rj_flux"], data["source_rj_flux"], data["source_rj_flux_error"], data["chi"])
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
            "SELECT {0} FROM {1};".format(
                ", ".join(column_names),
                table
            )
        )
        rows = cur.fetchall()
        # rows = [r[0] for r in cur.fetchall()]
        print("The number of entries: ", cur.rowcount)
        
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
