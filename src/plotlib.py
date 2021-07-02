#!/usr/bin/env python

"""plotlib.py: module is dedicated to all plot functions."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_damp_function(x, y, fname):
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time, mins.")
    ax.set_ylabel("Damping function")
    ax.plot(x, y, ls="--", lw=0.8, color="b")
    ax.set_ylim(0,1)
    ax.set_xlim(0,120)
    fig.savefig(fname, bbox_inches="tight")
    return

def plot_echoes(x, y, fname):
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time, mins.")
    ax.set_ylabel("#-GS")
    for i in y: 
        ax.plot(x, i, marker="o", ls="None", ms=1., color="b")
    ax.set_ylim(0,40)
    ax.set_xlim(0,120)
    fig.savefig(fname, bbox_inches="tight")
    return

def plot_fitdata(u, fname):
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time, mins.")
    ax.set_ylabel("#-GS")
    fmt = mdates.DateFormatter("%H")
    ax.xaxis.set_major_formatter(fmt)
    ax.plot(u.time, u.echoes, marker="o", ls="None", ms=1., color="b")
    ax.set_ylim(0,40)
    ax.set_xlim(u.time.tolist()[0],u.time.tolist()[-1])
    fig.savefig(fname, bbox_inches="tight")
    return


def plot_fit_data_with_scores(u, scores, fname):
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time, mins.")
    ax.set_ylabel("#-GS")
    fmt = mdates.DateFormatter("%H")
    ax.xaxis.set_major_formatter(fmt)
    ax.plot(u.time, u.echoes, marker="o", ls="None", ms=1., color="b", alpha=0.5)
    ax.set_ylim(0,40)
    ax = ax.twinx()
    ax.set_ylabel("score")
    ax.xaxis.set_major_formatter(fmt)
    ax.plot(u.time.tolist()[:-1], scores, marker="o", ls="None", ms=1., color="r", alpha=0.5)
    ax.set_xlim(u.time.tolist()[0],u.time.tolist()[-1])
    fig.savefig(fname, bbox_inches="tight")
    return
