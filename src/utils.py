#!/usr/bin/env python

"""utils.py: module is dedicated to all utility functions."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import pandas as pd
import glob
from pathlib import Path
import datetime as dt

def generate_stats(_dir_="../results/"):
    files = glob.glob(_dir_ + "*.csv")
    files.sort()
    u = pd.DataFrame()
    for f in files:
        u = pd.concat([u, pd.read_csv(f, parse_dates=["st", "et"])])
    print(u.head())
    u = u[(u.st>dt.datetime(2015,3,1)) & (u.st<dt.datetime(2015,4,1))]
    print(u[["st", "mprob", "jscore"]].values)
    fc = pd.read_csv("../goes_merged_flare_catalog.txt", parse_dates=["dt"])
    fc = fc[(fc.dt>dt.datetime(2015,3,1)) & (fc.dt<dt.datetime(2015,4,1)) & (fc.B_AVG > 1e-5)]
    print(fc[["dt", "class"]].values)
    return


if __name__ == "__main__":
    generate_stats()
    pass
