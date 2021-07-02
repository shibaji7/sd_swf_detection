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

import os
import numpy as np
import glob
import pandas as pd

import plotlib

def create_data_folder_structures(base="../data"):
    if not os.path.exists(base): os.system("mkdir " + base)
    if not os.path.exists(base + "/raw"): os.system("mkdir " + base + "/raw")
    if not os.path.exists(base + "/damp"): os.system("mkdir " + base + "/damp")
    if not os.path.exists(base + "/train"): os.system("mkdir " + base + "/train")
    if not os.path.exists(base + "/test"): os.system("mkdir " + base + "/test")
    if not os.path.exists(base + "/validation"): os.system("mkdir " + base + "/validation")
    return

def create_plot_folder_structures(base="../plots"):
    if not os.path.exists(base): os.system("mkdir " + base)
    if not os.path.exists(base + "/raw"): os.system("mkdir " + base + "/raw")
    if not os.path.exists(base + "/damp"): os.system("mkdir " + base + "/damp")
    if not os.path.exists(base + "/train"): os.system("mkdir " + base + "/train")
    if not os.path.exists(base + "/test"): os.system("mkdir " + base + "/test")
    if not os.path.exists(base + "/validation"): os.system("mkdir " + base + "/validation")
    return

def create_backscatter_echoes(base=15, beams=range(4,24), 
        noise_params={"sigma":1, "mean":0}, seed=0, dur=120):
    np.random.seed(seed)
    _o = np.empty((len(beams), dur))
    for _i, b in enumerate(beams):
        _o[_i, :] = np.random.normal(loc=base+noise_params["mean"]+(b/np.mean(beams)),
                scale=noise_params["sigma"], size=dur).astype(int) 
    return _o

def generate_backscatter_echoes(echo={"loc":20,"scale":5,"samples":20}, 
        noise={"loc":5,"scale":2,"samples":20}, seed=0, dur=120, beams=range(4,24), 
        base_dir="../data/raw/", fname="dat_%04d_%03d_%d.txt", clean=True, plot=False):
    if clean: os.system("rm -rf " + base_dir + "*")
    np.random.seed(seed)
    noise_params_distribution = np.random.gamma(noise["loc"], noise["scale"], noise["samples"])
    echo_params_distribution = np.random.normal(echo["loc"], echo["scale"], echo["samples"])
    i = 0
    for n in noise_params_distribution:
        for e in echo_params_distribution:
            o = create_backscatter_echoes(base=e, beams=beams,
                    noise_params={"sigma":n, "mean":0}, seed=seed, dur=dur)
            print(" Creating data for %d bsc. and %d noise."%(e,n))
            floc = base_dir + fname%(e,n,i)
            np.savetxt(floc, o)
            if plot: plotlib.plot_echoes(np.arange(dur), o, floc.replace("txt", "png").replace("/data/", "/plots/"))
            i += 1
    return

def create_damp_functions(flare_time_location=30, percentage_drop=1.,
        drop_rate=3., gain_rate=1e-1, dur=120, seed=0):
    np.random.seed(seed)
    fn_sech = lambda x, a0: percentage_drop*(2/(np.exp(a0*x) + np.exp(-a0*x)))
    xs = np.arange(0, flare_time_location+1) - flare_time_location
    ys = np.arange(1, dur-flare_time_location)
    drop = fn_sech(xs, drop_rate)
    recovery = fn_sech(ys, gain_rate)
    dmp = 1.-np.round(drop.tolist() + recovery.tolist(),2)
    return dmp

def generate_backscatter_damp_function(dur=120, seed=0, base_dir="../data/damp/", fname="dmp_dat_%03d_%.2f_%.2f_%.2f_%d.txt", clean=True,
        plot=False):
    if clean: os.system("rm -rf " + base_dir + "*")
    np.random.seed(seed)
    flare_time_location_distribution = np.random.uniform(10, 90, size=10).astype(int)
    percentage_drop_ditsribution = np.round(np.random.uniform(0.5, 2., size=5), 2)
    drop_rate_distribution = np.round(np.random.normal(3, .5, size=5), 2)
    gain_rate_distribution = np.round(np.random.uniform(5e-2, 1e-1, size=5), 2)
    i = 0
    for f in flare_time_location_distribution:
        for p in percentage_drop_ditsribution:
            for dr in drop_rate_distribution:
                for gr in gain_rate_distribution:
                    dmp = create_damp_functions(f, p, dr, gr, dur, seed)
                    print(" Creating data for %d flare, %.2f percentage rate, %.2f drop rate, and %.2f gain rate."%(f,p,dr,gr))
                    floc = base_dir + fname%(f,p,dr,gr,i)
                    np.savetxt(floc, dmp)
                    if plot: plotlib.plot_damp_function(np.arange(dur), dmp, floc.replace("txt", "png").replace("/data/", "/plots/"))
    raw_files = glob.glob("../data/raw/*.txt")
    damp_files = glob.glob("../data/damp/*.txt")
    print(" Total combinations: %d [%d X %d]"%(len(damp_files)*len(raw_files), len(raw_files), len(damp_files)))
    return

def create_train_test_dataset(raw_dir="../data/raw/", damp_dir="../data/damp/", train_dir="../data/train/", 
        test_dir="../data/test/", validation_dir="../data/validation/"):
    os.system("rm -rf " + train_dir + "*")
    os.system("rm -rf " + test_dir + "*")
    os.system("rm -rf " + validation_dir + "*")
    raw_files = glob.glob(raw_dir + "*.txt")
    damp_files = glob.glob(damp_dir + "*.txt")
    print(" Total combinations: %d [%d X %d]"%(len(damp_files)*len(raw_files), len(raw_files), len(damp_files)))
    dmap = {"train": [], "test": [], "validation": []}
    idx = 0
    for rfile in raw_files:
        r_dat = np.loadtxt(rfile)
        for dfile in damp_files:
            if np.random.uniform(0,1) > 0.5:
                d_dat = np.loadtxt(dfile)
                o = np.multiply(r_dat, d_dat.T)
                o[o <= 0.] = 0.
                o = np.mean(o, axis=0)
                fname = rfile.split("/")[-1]
                params = dfile.split("/")[-1]
                params = "_" + params.replace("_dat", "")
                fname = fname.replace(".txt", params)
                params = params.replace(".txt","").split("_")
                end = np.argmin(abs(d_dat[int(params[2]):]-0.92)) + int(params[2])
                pobj = {"start": int(params[2]), "end": end, "peak": float(params[3]), "swf_flag": True}
            else:
                o = np.mean(np.copy(r_dat), axis=0)
                fname = rfile.split("/")[-1]
                params = "_dmp_000_0.00_0.00_0.00.txt"
                fname = fname.replace(".txt", params)
                pobj = {"start": -1, "end": -1, "peak": 0., "swf_flag": False}
            u = np.argmax(np.random.multinomial(1, [0.8, 0.1, 0.1], size=1)[0])
            if u == 0: key, _dir_ = "train", train_dir
            elif u == 1: key, _dir_ = "test", test_dir
            elif u == 2: key, _dir_ = "validation", validation_dir
            dmap[key].append(pobj)
            floc = _dir_ + "%06d.txt"%idx
            np.savetxt(floc, o)
            idx += 1
    train = pd.DataFrame.from_records(dmap["train"])
    test = pd.DataFrame.from_records(dmap["validation"])
    validation = pd.DataFrame.from_records(dmap["validation"])
    train.to_csv("../data/train_source.csv", header=True, index=False)
    test.to_csv("../data/test_source.csv", header=True, index=False)
    validation.to_csv("../data/validation_source.csv", header=True, index=False)
    return

if __name__ == "__main__":
    cases = [0,1,2,3,4]
    for case in cases:
        if case==0: create_data_folder_structures()
        if case==1: create_plot_folder_structures()
        if case==2: generate_backscatter_echoes()
        if case==3: generate_backscatter_damp_function()
        if case==4: create_train_test_dataset()
    os.system("rm -rf *log*")
    os.system("rm -rf __pycache__")
    pass
