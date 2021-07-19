#!/usr/bin/env python

"""detector.py: module is dedicated to all detector functions."""

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
from scipy.stats import median_absolute_deviation
from scipy.signal import butter, lfilter
import traceback

from get_fit_data import get_date_by_dates
import plotlib

import json
import uuid

with open("prop.json", "r") as f: properties = json.load(f)

def smooth(x, window_len=51, window="hanning"):
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

class Detector(object):
    """ Genertic detector class to be inhereited by all other scheme """

    def __init__(self, dates, rad, kind, idx, plot=True, verbose=False):
        self.dates = dates
        self.rad = rad
        self.kind = kind
        self.plot = plot
        self.verbose = verbose
        self.uid = idx

        self.data_fname = properties["data_fname"]%(dates[0].strftime("%Y-%m-%d"), rad)
        self.result_fname = (properties["result_fname"]%(dates[0].strftime("%Y-%m-%d"), rad, kind)).format(id="%04d"%self.uid)
        self.sza_th = properties["sza_th"]
        self.smoothing_window = properties["smoothing_window"]
        self.dur = properties["dur"]

        self.records = []
        self.scores = []
        self.times = []
        return

    def smooth(self, x, window_len=51, window="hanning"):
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

    def get_parse_radar_data(self):
        parsed = True
        try:
            if not os.path.exists(self.data_fname):
                self.o = get_date_by_dates(self.rad, self.dates)
                print(" Length:", len(self.o))
                self.o = self.o[self.o.sza <= self.sza_th]
                print(" Modified length:", len(self.o))
                self.o.to_csv(self.data_fname, header=True, index=False)
            else: self.o = pd.read_csv(self.data_fname, parse_dates=["time"])
            self.beams = self.o.beams.unique()
        except:
            parsed = False
            if self.verbose: traceback.print_exc()
        return parsed

    def run(self):
        """ Main function call """
        if not os.path.exists(self.result_fname):
            if self.get_parse_radar_data():
                rec = []
                load = True
                for b in self.beams:
                    x = self.o[self.o.beams==b]
                    x = x.set_index("time").resample(properties["resample_time"]).median().reset_index()\
                            .interpolate(limit=properties["max_intrepolate_gap"])
                    x.echoes = self.smooth(np.array(x.echoes), self.smoothing_window)
                    out = self.scheme(x, self.dates[0], self.dates[1], b, dur=self.dur, load=True)
                    rec.extend(out)
                    load = False
                self.df = pd.DataFrame.from_records(rec)
                self.invoke_parser()
        return

    def scheme(self):
        """ Needs to be overloaded """
        return

    def invoke_parser(self):
        """ May need to be overloaded """
        self.records = []
        if len(self.df) > 0:
            stats = self.df.groupby(by=["st"]).agg({"prob":[np.median, "count", median_absolute_deviation],
                "qprob":[np.median, "count", median_absolute_deviation],
                }).reset_index()
            L = 1./len(self.beams)
            for i in range(len(stats)):
                st, et = stats.iloc[i]["st"].tolist()[0], stats.iloc[i]["st"].tolist()[0] + dt.timedelta(minutes=120)
                mprob, cprob = stats.iloc[i]["prob"]["median"], stats.iloc[i]["prob"]["count"]*L
                mad = stats.iloc[i]["prob"]["median_absolute_deviation"]
                jscore = -10*np.log10(stats.iloc[i]["prob"]["median_absolute_deviation"]) if mad > 0. else -1
                q_mprob, q_cprob = stats.iloc[i]["qprob"]["median"], stats.iloc[i]["qprob"]["count"]*L
                mad = stats.iloc[i]["qprob"]["median_absolute_deviation"]
                q_jscore = -10*np.log10(stats.iloc[i]["qprob"]["median_absolute_deviation"]) if mad > 0. else -1
                self.records.append({"st": st, "et": et, "rad": self.rad, "mprob": mprob, "jscore": jscore, "cprob": cprob,
                    "qmprob": q_mprob, "qjscore": q_jscore, "qcprob": q_cprob})
                print(" Chance of SWF between (%s, %s) observed by %s radar are (%.2f, %.2f, %.2f) (%.2f, %.2f, %.2f)"%\
                        (st.strftime("%Y-%m-%d %H:%M"), et.strftime("%Y-%m-%d %H:%M"), self.rad.upper(), mprob, cprob, 
                            jscore, q_mprob, q_cprob, q_jscore))
        return

    def save(self):
        if len(self.records)>0: pd.DataFrame.from_records(self.records).to_csv(self.result_fname, header=True, index=False)
        return

    def get_qprob(self, x):
        q = qprob = 1./(1.+np.exp(((np.quantile(x, 0.1)-np.quantile(x, 0.9))/np.quantile(x, 0.1))))
        return q


class ZScore(Detector):
    """ Whitaker-Hayes algorithm: Z Score based method """

    def __init__(self,  dates, rad, idx, plot=True):
        super().__init__(dates, rad, "zscore", idx, plot)
        self.threshold = properties["z_score_threshold"]
        return

    def modified_z_score(self, intensity):
        median_int = np.median(intensity)
        mad_int = np.median([np.abs(intensity - median_int)])
        modified_z_scores = 0.6745 * (intensity - median_int) / mad_int
        return modified_z_scores

    def scheme(self, x, start, end, b, dur, load):
        """ Overloaded """
        st, et = start, start + dt.timedelta(minutes=dur)
        coll = []
        while et <= end:
            u = x[(x.time >= st) & (x.time < et)]
            if len(u) == dur and not u.isnull().values.any():
                qprob = self.get_qprob(u.echoes)
                delta_intensity = []
                intensity = np.array(u.echoes)
                for i in np.arange(len(u)-1):
                    dist = intensity[i+1] - intensity[i]
                    delta_intensity.append(dist)
                delta_int = np.array(delta_intensity)
                scores = np.array(self.modified_z_score(delta_int).tolist() + [0])
                if load:
                    #scores[scores<=np.quantile(scores,0.1)] = np.nan
                    #scores[scores>=np.quantile(scores,0.9)] = np.nan
                    self.scores.extend(scores.tolist())
                    self.times.extend(u.time.tolist())
                scores[scores > 0.] = 0.
                n_spikes = np.count_nonzero(np.abs(np.array(scores)) > self.threshold)
                p = 0.
                if self.plot:
                    self.plot_fname = "../plots/rad_summary_%s_%s_%s_%02d.png"%(st.strftime("%Y-%m-%d-%H-%M"), 
                            et.strftime("%Y-%m-%d-%H-%M"), self.rad, b)
                    plotlib.plot_fit_data_with_scores(u, scores, self.plot_fname)
                if n_spikes > 0: p = 1./(1.+np.exp(scores.min()-self.threshold))
                obj = {"beam":b, "st": st, "et": et, "prob": p, "qprob":qprob}
                coll.append(obj)
            st = et
            et += dt.timedelta(minutes=dur)
        return coll

class CascadingNEO(Detector):
    """ Nonlinear Energy Operator: NEO, Implemented from 'Holleman2011.Chapter.SpikeDetection_Characterization' """

    def __init__(self,  dates, rad, idx, plot=True):
        super().__init__(dates, rad, "neo", idx, plot)
        self.threshold = properties["neo_threshold"]
        self.neo_order = properties["neo_order"]
        return

    def neo(self, x):
        """
            neo(x): diff(x,2)-x.diff(x,1)
            Implemented from "Holleman2011.Chapter.SpikeDetection_Characterization"
        """
        y = np.gradient(x,edge_order=1)**2 - (np.gradient(x,edge_order=2)*x)
        return y

    def scheme(self, x, start, end, b, dur, load):
        """ Overloaded """
        st, et = start, start + dt.timedelta(minutes=dur)
        coll = []
        while et <= end:
            u = x[(x.time >= st) & (x.time < et)]
            if len(u) == dur and not u.isnull().values.any():
                qprob = self.get_qprob(u.echoes)
                scores = np.array(u.echoes)
                for _i in range(self.neo_order):
                    scores = self.neo(scores)
                n_spikes = np.count_nonzero(np.abs(np.array(scores)) > self.threshold)
                if load:
                    #scores[scores<=np.quantile(scores,0.1)] = np.nan
                    #scores[scores>=np.quantile(scores,0.9)] = np.nan
                    self.scores.extend((scores/(10**self.neo_order)))
                    self.times.extend(u.time.tolist())
                p = 0.
                if self.plot:
                    self.plot_fname = "../plots/rad_summary_%s_%s_%s_%02d.png"%(st.strftime("%Y-%m-%d-%H-%M"),
                            et.strftime("%Y-%m-%d-%H-%M"), self.rad, b)
                    plotlib.plot_fit_data_with_scores(u, scores, self.plot_fname)
                if n_spikes > 0: p = 1./(1.+np.exp(-((np.abs(scores).max()/(10**self.neo_order))-self.threshold)))
                obj = {"beam":b, "st": st, "et": et, "prob": p, "qprob": qprob}
                coll.append(obj)
            st = et
            et += dt.timedelta(minutes=dur)
        return coll


def algorithm_runner_helper(dates, rad, kind, idx, plot=True, save=True):
    if kind == "zscore": method = ZScore(dates, rad, idx, plot=plot)
    if kind == "neo": method = CascadingNEO(dates, rad, idx, plot=plot)
    method.run()
    if save: method.save()
    return method

def run_parallel_procs(dates, rad, kind="zscore", plot=True, save=True, idx=False, procs=8):
    from multiprocessing import Pool
    from functools import partial
    dt_args = []
    sdate, edate = dates[0], dates[1]
    if idx:
        idx = str(uuid.uuid1())
        if save:
            result_fname = (properties["result_fname"]%(dates[0].strftime("%Y-%m-%d"), rad, kind)).format(id=self.uid)
            _dir = "/".join(result_fname.split("/")[:-1])
            if not os.path.exists(_dir):
                os.system("mkdir "+_dir)
                with open(_dir+"/prop.json", "w") as f: f.write(json.dumps(properties, indent=4))
    else: 
        properties["result_fname"] = "../results/%s_%s_%s.csv"
        idx=""
    pool = Pool(processes=procs)
    while sdate <= edate:
        dt_args.append([sdate, sdate + dt.timedelta(1)])
        sdate = sdate + dt.timedelta(1)
    pool.map(partial(algorithm_runner_helper, rad=rad, kind=kind, idx=idx, plot=plot, save=save), dt_args)
    return

def run_batch_for_radar(rad, dates, kind, prop, plot=False, save=True, procs=8):
    from multiprocessing import Pool
    from functools import partial
    dt_args = []
    idx = prop["idx"]
    for p in prop.keys():
        properties[p] = prop[p]
    if save:
        result_fname = (properties["result_fname"]%(dates[0].strftime("%Y-%m-%d"), rad, kind)).format(id="%04d"%idx)
        _dir = "/".join(result_fname.split("/")[:-1])
        if not os.path.exists(_dir):
            os.system("mkdir "+_dir)
            with open(_dir+"/prop.json", "w") as f: f.write(json.dumps(properties, indent=4))
    pool = Pool(processes=procs)
    for d in dates:
        dt_args.append([d, d + dt.timedelta(1)])
    pool.map(partial(algorithm_runner_helper, rad=rad, kind=kind, idx=idx, plot=plot, save=save), dt_args)
    return


if __name__ == "__main__":
    dates = [dt.datetime(2015,5,5), dt.datetime(2015,5,5)]
    run_parallel_procs(dates, "bks", "neo", True, False)
    #run_parallel_procs(dates, "fhe", "neo", False, True)
    #run_parallel_procs(dates, "kap", "neo", False, True)
    #run_parallel_procs(dates, "bks", "zscore", False, True)
    #run_parallel_procs(dates, "fhe", "zscore", False, True)
    #run_parallel_procs(dates, "kap", "zscore", False, True)
    pass
