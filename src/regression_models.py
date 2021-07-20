#!/usr/bin/env python

"""regression_models.py: module is dedicated to produce the regression models."""

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
import pandas as pd
import datetime as dt
import boxcar_filter as box

class Regressor(object):
    """ Genertic regressor class to be inhereited by all other models """

    def __init__(self, date, rad, base="../"):
        self.rad = rad
        self.date = date
        self.data_fname = base + "regressor/data/%s_%s"%(rad, date.strftime(""))
        return

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

def fetch_goes_data(dates, sat="g15"):
    def last_day_of_month(ad):
        next_month = ad.replace(day=28) + dt.timedelta(days=4)
        return next_month - dt.timedelta(days=next_month.day)
    satf = sat.replace("g", "goes")
    d0, d1 = dates[0].replace(day=1), last_day_of_month(dates[1])
    url = "https://satdat.ngdc.noaa.gov/sem/goes/data/avg/%04d/%02d/%s/csv/%s_xrs_1m_{ds}_{de}.csv"%(d0.year, d0.month, satf, sat)
    url = url.format(ds = d0.strftime("%Y%m%d"), de=d1.strftime("%Y%m%d"))
    x = pd.read_csv(url, parse_dates=["time_tag"], skiprows=167)
    x = x[(x.time_tag>=dates[0]) & (x.time_tag<dates[1])]
    return x

def maintain_event_list(base="../", event_list_files=["regressor/event_list.csv"], clean=True):
    date_list, radar_list, dur_list = [], [], []
    for f in event_list_files:
        events = pd.read_csv(base + f, parse_dates=["date"])
        events = events[events.durations.notnull()]
        for _, e in events.iterrows():
            dn, radars, dur = e["date"], e["radars"].split("-"), e["durations"]
            for r in radars:
                print("Event - ", dn, r, dur)
                date_list.append(dn)
                radar_list.append(r)
                dur_list.append(dur)
    x = pd.DataFrame()
    x["rad"], x["date"], x["durations"], x["rad_fnames"], x["goes_fnames"] = radar_list, date_list, dur_list,\
            [base + "regressor/data/rad_%04d.csv"%t for t in range(len(dur_list))],\
            [base + "regressor/data/goes_%04d.csv"%t for t in range(len(dur_list))]
    x.to_csv(base + "regressor/events.csv", header=True, index=False)

    for i, r in x.iterrows():
        if clean: os.system("rm -rf " + r["rad_fnames"] + " " + r["goes_fnames"])
        dates = [r["date"] - dt.timedelta(minutes=int(r["durations"]/2)),
                r["date"] + dt.timedelta(minutes=r["durations"])]
        rad = r["rad"]
        if not os.path.exists(r["rad_fnames"]):
            g = fetch_goes_data(dates, "g15")
            g.to_csv(r["goes_fnames"], header=True, index=False)
            u = box.get_med_filt_data_by_dates(rad, dates, low=20, high=60)
            u.to_csv(r["rad_fnames"], header=True, index=False)
    return

def fit_indp_models(o):
    x = np.log10(o[["B_AVG","A_AVG"]].values)
    y = o[["echoes"]].values
    import statsmodels.api as sm
    x = sm.add_constant(x)
    return

def create_fit_dataset(base="../", fname="regressor/events.csv"):
    events = pd.read_csv(base + fname, parse_dates=["date"])
    for i, e in events.iterrows():
        gfile, rfile = e["goes_fnames"], e["rad_fnames"]
        g = pd.read_csv(gfile, parse_dates=["time_tag"])
        g = g[(g.A_QUAL_FLAG==0) & (g.B_QUAL_FLAG==0)]
        g = g[["time_tag", "B_AVG", "A_AVG"]]
        g = g.rename(columns={"time_tag":"time"})
        g = g.set_index("time").resample("1s").interpolate()
        u = pd.read_csv(rfile, parse_dates=["time"])
        u.time = [t.replace(microsecond=0) for t in u.time]
        u = u.set_index("time").resample("1s").interpolate()
        u.beams, u.tfreq, u.echoes = np.rint(u.beams).astype(int), np.rint(u.tfreq).astype(int), np.rint(u.echoes).astype(int)
        g = g.iloc[60:(len(u)+60)]
        m = pd.merge(u,g, how="inner", left_index=True, right_index=True).reset_index()
        m.to_csv(gfile.replace("goes_", ""), header=True, index=False)
        fit_indp_models(m)
    return

if __name__ == "__main__":
    #maintain_event_list()
    create_fit_dataset()
    pass
