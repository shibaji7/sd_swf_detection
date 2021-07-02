#!/usr/bin/env python

"""binary_classification.py: module is dedicated to all binary classification functions."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import datetime as dt
import numpy as np
import pandas as pd
import glob

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from joblib import dump, load

from get_fit_data import get_date_by_dates
import plotlib
from scipy.stats import median_absolute_deviation


def get_local_data_from_disk_data():
    train_labels = np.array(pd.read_csv("../data/train_source.csv")["swf_flag"])
    test_labels = np.array(pd.read_csv("../data/test_source.csv")["swf_flag"])
    validation_labels = np.array(pd.read_csv("../data/validation_source.csv")["swf_flag"])
    labels = (train_labels, test_labels, validation_labels)
    features = []
    for dirs in ["../data/train/*.txt", "../data/test/*.txt", "../data/validation/*.txt"]:
        dx = []
        files = glob.glob(dirs)
        files.sort()
        for f in files:
            print(" Load:", f)
            o = np.loadtxt(f)
            dx.append(o)
        features.append(np.array(dx))
    return labels, tuple(features)

def reshape_labels_features(labels, features):
    lab, fea = [], []
    for l, f in zip(labels, features):
        if len(l) >= f.shape[0]: lab.append(l[:len(f)]); fea.append(f)
        if len(l) < f.shape[0]: lab.append(l); fea.append(f[:len(l)])
    return tuple(lab), tuple(fea)

def train_base_model(clean=False, kind="random_forest", 
        params={"max_depth":10, "random_state":0}, save_model=True, load_model=True, verbose=True):
    if clean: os.system("rm -rf ../data/prep/")
    if not os.path.exists("../data/prep/"):
        labels, features = get_local_data_from_disk_data()
        os.system("mkdir ../data/prep/")
        tags = ["train", "test", "validation"]
        for l, f, t in zip(labels, features, tags):
            np.savetxt("../data/prep/%s_labels.txt"%t, l)
            np.savetxt("../data/prep/%s_features.txt"%t, f)
    else:
        labels = (np.loadtxt("../data/prep/train_labels.txt"), np.loadtxt("../data/prep/test_labels.txt"),
                np.loadtxt("../data/prep/validation_labels.txt"))
        features = (np.loadtxt("../data/prep/train_features.txt"), np.loadtxt("../data/prep/test_features.txt"),
                np.loadtxt("../data/prep/validation_features.txt"))
    labels, features = reshape_labels_features(labels, features)
    fname = "../model/%s.joblib"%kind
    if verbose: print(" Loading model...")
    if load_model and os.path.exists(fname): model = load(fname)
    else:
        if verbose: print(" Fitting model...")
        if kind: model = RandomForestClassifier(max_depth=params["max_depth"], random_state=params["random_state"])
        model.fit(features[0], labels[0])
    if verbose: print(" Accuracy:", model.score(features[2], labels[2]))
    if save_model: dump(model, fname) 
    return

def run_detection_scheme(model, x, start, end, b, dur=120, sep=30, verbose=False):
    st, et = start, start + dt.timedelta(minutes=dur) 
    coll = []
    while et <= end:
        u = x[(x.time >= st) & (x.time < et)]
        if len(u) == dur and not u.isnull().values.any():
            pr_d = model.predict_proba(np.array(u.echoes).reshape(1, dur))[0,1]
            if pr_d > 0.8:
                if verbose: print(" Prob. of SWF in this (%s, %s) window for beam %02d is %.2f"%(st.strftime("%H:%M"), 
                    et.strftime("%H:%M"), b, pr_d))
                coll.append({"beam":b, "st": st, "et": et, "prob": pr_d})
            pass
        st = et
        et += dt.timedelta(minutes=dur)
    return coll

def run_finder_scheme(x, start, end, b, dur=120, sep=30, verbose=False):
    st, et = start, start + dt.timedelta(minutes=dur)
    coll = []
    while et <= end:
        u = x[(x.time >= st) & (x.time < et)]
        if len(u) == dur and not u.isnull().values.any():
            pr_d = model.predict_proba(np.array(u.echoes).reshape(1, dur))[0,1]
            if pr_d > 0.8:
                if verbose: print(" Prob. of SWF in this (%s, %s) window for beam %02d is %.2f"%(st.strftime("%H:%M"),
                    et.strftime("%H:%M"), b, pr_d))
                coll.append({"beam":b, "st": st, "et": et, "prob": pr_d})
            pass
        st = et
        et += dt.timedelta(minutes=dur)
    return coll

def smooth(x,window_len=51,window="hanning"):
    if x.ndim != 1: raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len: raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3: return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]: raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == "flat": w = numpy.ones(window_len,"d")
    else: w = eval("np."+window+"(window_len)")
    y = np.convolve(w/w.sum(),s,mode="valid")
    d = window_len - 1
    y = y[int(d/2):-int(d/2)]
    return y

def run_model_on_SD_data_by_beams(i_dates, rad, kind="random_forest"):
    fname = "../results/%s.csv"%(i_dates[0].strftime("%Y-%m-%d"))
    if not os.path.exists(fname):
        print(" Loading model...")
        fname = "../model/%s.joblib"%kind
        model = load(fname)
        sdate, edate = i_dates[0], i_dates[1]
        records = []
        while sdate < edate:
            try:
                print(" Processing for date:",sdate.strftime("%Y-%m-%d"))
                dates = [sdate, sdate+dt.timedelta(1)]
                o = get_date_by_dates(rad, dates)
                print(" Length:", len(o))
                o = o[o.sza <= 95.]
                print(" Modified length:", len(o))
                beams = o.beams.unique()
                rec = []
                for b in beams:
                    x = o[o.beams==b]
                    x = x.set_index("time").resample("1T").median().reset_index().interpolate(limit=2)
                    x.echoes = smooth(np.array(x.echoes), 51)
                    fname = "../plots/rad_summary_%s_%02d.png"%(dates[0].strftime("%Y-%m-%d"), b)
                    plotlib.plot_fitdata(x, fname)
                    rec.extend(run_detection_scheme(model, x, dates[0], dates[1], b, dur=120, sep=30))
                rec = pd.DataFrame.from_records(rec)
                if len(rec) > 0:
                    stats = rec.groupby(by=["st"]).agg({"prob":[np.median, "count"]}).reset_index()
                    joint_prob = rec.groupby(by=["st"], as_index=False).prod().reset_index()
                    L = 1./len(beams)
                    for i in range(len(stats)):
                        st, et = stats.iloc[i]["st"].tolist()[0], stats.iloc[i]["st"].tolist()[0] + dt.timedelta(minutes=120)
                        mprob, jprob = stats.iloc[i]["prob"]["median"]*stats.iloc[i]["prob"]["count"]*L, -10*np.log10(joint_prob.iloc[i]["prob"])
                        records.append({"st": st, "et": et, "rad": rad, "mprob": mprob, "jprob": jprob})
            except: print(" System exception...")
            sdate = dates[1]
        print("\n\n\n")
        for r in records:
            st, mt, mprob, jprob, rad = r["st"], r["et"], r["mprob"], r["jprob"], r["rad"]
            print(" Chance of SWF between (%s, %s) observed by %s radar is (%.2f, %.2f)"%(st.strftime("%Y-%m-%d %H:%M"),
                et.strftime("%Y-%m-%d %H:%M"), rad.upper(), mprob, jprob))
        pd.DataFrame.from_records(records).to_csv("../results/%s.csv"%(i_dates[0].strftime("%Y-%m-%d")), header=True, index=False)
    return

def run_model_on_SD_data(i_dates, rad, kind="random_forest"):
    fname = "../results/%s_by_rad.csv"%(i_dates[0].strftime("%Y-%m-%d"))
    if not os.path.exists(fname):
        print(" Loading model...")
        fname = "../model/%s.joblib"%kind
        model = load(fname)
        sdate, edate = i_dates[0], i_dates[1]
        records = []
        while sdate < edate:
            try:
                print(" Processing for date:",sdate.strftime("%Y-%m-%d"))
                dates = [sdate, sdate+dt.timedelta(1)]
                o = get_date_by_dates(rad, dates)
                print(" Length:", len(o))
                o = o[o.sza <= 95.]
                print(" Modified length:", len(o))
                beams = o.beams.unique()
                rec = []
                o = o.set_index("time").resample("1T").median().reset_index().interpolate(limit=2)
                o.echoes = smooth(np.array(o.echoes), 51)
                fname = "../plots/rad_summary_%s.png"%(dates[0].strftime("%Y-%m-%d"))
                plotlib.plot_fitdata(o, fname)
                rec.extend(run_finder_scheme(model, o, dates[0], dates[1], -1, dur=120, sep=30))
                rec = pd.DataFrame.from_records(rec)
                if len(rec) > 0:
                    stats = rec.groupby(by=["st"]).agg({"prob":[np.median, "count"]}).reset_index()
                    joint_prob = rec.groupby(by=["st"], as_index=False).prod().reset_index()
                    L = 1./len(beams)
                    for i in range(len(stats)):
                        st, et = stats.iloc[i]["st"].tolist()[0], stats.iloc[i]["st"].tolist()[0] + dt.timedelta(minutes=120)
                        mprob, jprob = stats.iloc[i]["prob"]["median"]*stats.iloc[i]["prob"]["count"]*L, -10*np.log10(joint_prob.iloc[i]["prob"])
                        records.append({"st": st, "et": et, "rad": rad, "mprob": mprob, "jprob": jprob})
            except: 
                print(" System exception...")
                import traceback
                traceback.print_exc()
            sdate = dates[1]
        print("\n\n\n")
        for r in records:
            st, mt, mprob, jprob, rad = r["st"], r["et"], r["mprob"], r["jprob"], r["rad"]
            print(" Chance of SWF between (%s, %s) observed by %s radar is (%.2f, %.2f)"%(st.strftime("%Y-%m-%d %H:%M"),
                et.strftime("%Y-%m-%d %H:%M"), rad.upper(), mprob, jprob))
        if len(records) > 0:
            pd.DataFrame.from_records(records).to_csv("../results/%s_by_rad.csv"%(i_dates[0].strftime("%Y-%m-%d")), header=True, index=False)
    return


def run_parallel_procs(dates, rad, kind="random_forest", procs=8):
    from multiprocessing import Pool
    from functools import partial
    dt_args = []
    sdate, edate = dates[0], dates[1]
    pool = Pool(processes=procs)
    while sdate < edate:
        print(sdate, edate)
        dt_args.append([sdate, sdate + dt.timedelta(1)])
        sdate = sdate + dt.timedelta(1)
    pool.map(partial(run_model_on_SD_data, rad=rad, kind=kind), dt_args)
    return

if __name__ == "__main__":
    run_parallel_procs([dt.datetime(2015,3,1), dt.datetime(2015,4,1)], "bks")
    pass
