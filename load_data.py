'''
Load zipped data frame file formats.
'''
import sqlite3
import zipfile

import pandas as pd


def read_zipped_csv(zipfilename, filename):
    """
    Open up a zipfile, and read out csv data from the speficied filename
    into a pandas dataframe.

    Parameters
    ----------
    zipfilename:
        Source zip file on disk.
    filename:
        File within the zip containing the data

    Returns
    -------
    A Pandas DataFrame.
    """
    # target path, wrapped in an object
    with zipfile.ZipFile(zipfilename) as archive:
        with archive.open(filename) as file:
            return pd.read_csv(file)


def read_zipped_sqlite(zipfilename, filename, query):
    """
    Open up a zipfile, and read out table data from the speficied filename
    into a pandas dataframe.

    Parameters
    ----------
    zipfilename:
        Source zip file on disk.
    filename:
        File within the zip containing the data
    tablename:
        Read this entire table.

    Returns
    -------
    A Pandas DataFrame.
    """
    # target path, wrapped in an object
    with zipfile.ZipFile(zipfilename) as archive:
        archive.extract(filename)
        with sqlite3.connect(filename) as connection:
            return pd.read_sql(query, connection)
