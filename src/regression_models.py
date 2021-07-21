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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os
import numpy as np
import pandas as pd
import datetime as dt

import statsmodels.formula.api as smf
from statsmodels.iolib.smpickle import load_pickle
from statsmodels.tools.eval_measures import rmse
from pysolar.solar import get_altitude

import boxcar_filter as box

FORMULAS = {
            "echoes": "np.log10(echoes+.1) ~ (B_AVG + A_AVG)*(np.power(np.cos(np.deg2rad(sza)),{gamma}))/np.power(tfreq,{delta})",
            "absp": "np.log10(absp+.1) ~ (B_AVG + A_AVG)*(np.power(np.cos(np.deg2rad(sza)),{gamma}))/np.power(tfreq,{delta})"
           }

def get_gridded_parameters(q, xparam="lon", yparam="lat", zparam="mean"):
    plotParamDF = q[ [xparam, yparam, zparam] ]
    plotParamDF[xparam] = plotParamDF[xparam].tolist()
    plotParamDF[yparam] = plotParamDF[yparam].tolist()
    plotParamDF = plotParamDF.groupby( [xparam, yparam] ).mean().reset_index()
    plotParamDF = plotParamDF[ [xparam, yparam, zparam] ].pivot( xparam, yparam )
    x = plotParamDF.index.values
    y = plotParamDF.columns.levels[1].values
    X, Y  = np.meshgrid( x, y )
    # Mask the nan values! pcolormesh can't handle them well!
    Z = np.ma.masked_where(
            np.isnan(plotParamDF[zparam].values),
            plotParamDF[zparam].values)
    return X,Y,Z

def plot_global_ditribution(o, fname, d):
    fig = plt.figure(dpi=120, figsize=(6,3))
    x, y, e = get_gridded_parameters(o)
    x, y, sza = get_gridded_parameters(o, zparam="sza")
    e[sza>90.] = np.nan
    ax = fig.add_subplot(111)
    c = ax.pcolormesh(x, y, e.T, cmap="Reds", vmin=0, vmax=50)
    fig.colorbar(c, ax=ax)
    CS = ax.contour(x, y, sza.T)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.text(0.01, 1.05, r"$f_0=%.1f\times 10^{6}$Hz"%o.iloc[0].tfreq, ha="left", va="center", transform=ax.transAxes)
    ax.text(0.99, 1.05, d.strftime("%Y.%m.%d %H.%M"), ha="right", va="center", transform=ax.transAxes)
    fig.savefig(fname, bbox_inches="tight")
    plt.close()
    return

def get_sza(geos):
    d = geos[2].replace(tzinfo=dt.timezone.utc)
    sza = 90.-get_altitude(geos[0], geos[1], d)
    return sza

def create_global_distribution(dates, freqs=[12.], base="../", ci=0.5):
    ls = load_pickle(base + "regressor/model.wls.pickle")
    N = 181
    lat, lon = np.linspace(-90, 90, N), np.linspace(-180, 180, N)
    lats, lons = np.meshgrid(lat, lon)
    x = pd.DataFrame()
    x["lat"], x["lon"] = lats.ravel(), lons.ravel()
    for d in dates:
        x["geolocate"] = [(la, lo, d) for la,lo in zip(lats.ravel(), lons.ravel())]
        x["sza"] = x.geolocate.apply(lambda a: get_sza(a))
        for f in freqs:
            x["tfreq"] = [f]*len(x)
            g = fetch_goes_data([d,d+dt.timedelta(minutes=1)], sat="g15").iloc[0]
            x["B_AVG"] = [g.B_AVG]*len(x)
            x["A_AVG"] = [g.A_AVG]*len(x)
            ypred = ((10**ls.get_prediction(x).summary_frame(alpha=1-ci))-.1)[["mean", "obs_ci_lower", "obs_ci_upper"]]
            o = pd.merge(x, ypred, how="inner", left_index=True, right_index=True).reset_index()
            plot_global_ditribution(o, base+"regressor/figures/%s_%.1f.png"%(d.strftime("%Y%m%d.%H%M"),f), d)
    return

def fit_global_absorption_models(base="../", fname="regressor/events.csv", check_params=False):
    x = pd.DataFrame()
    events = pd.read_csv(base + fname, parse_dates=["date"])
    for i, e in events.iterrows():
        gfile = base + e["goes_fnames"]
        pfname = gfile.replace("data/goes_", "proc/")
        x = pd.concat([x, pd.read_csv(pfname, parse_dates=["time"])])
    print(" Dataset - \n", x.head())
    # Tune gamma delta
    if check_params:
        Gamma, Delta, RMSE = [], [], []
        for gm in np.linspace(.1,1.5,15):
            for dl in np.linspace(1,2,11):
                gm, dl = np.round(gm, 1), np.round(dl, 1)
                f = FORMULAS["absp"].format(gamma=gm, delta=dl)
                model = smf.wls(formula=f, data=x, weights=1/(x.echoes+.1))
                results = model.fit()
                ypred = results.predict()
                print("Gamma, Delta, RMSE - ", gm, dl, rmse(x.echoes, ypred))
                Gamma.append(gm)
                Delta.append(dl)
                RMSE.append(rmse(x.echoes, ypred))
        du = pd.DataFrame()
        du["Gamma"], du["Delta"], du["RMSE"] = Gamma, Delta, RMSE
        global_gamma, global_delta = du.iloc[du.RMSE.argmin()].Gamma*10, du.iloc[du.RMSE.argmin()].Delta
    else: global_gamma, global_delta = 1, 2
    # Fitting global model
    print(" Gamma - Delta with minimum RMSE: ", global_gamma, global_delta)
    f = FORMULAS["absp"].format(gamma=global_gamma, delta=global_delta)
    print(" Formula - ", f)
    model = smf.wls(formula=f, data=x, weights=1/(x.echoes+0.1))
    model.fit().save(base + "regressor/model.wls.pickle")
    model = smf.ols(formula=f, data=x)
    model.fit().save(base + "regressor/model.ols.pickle")
    return

def generate_indp_fitted_plots(key, base="../", fname="regressor/events.csv", ci=0.5):
    events = pd.read_csv(base + fname, parse_dates=["date"])
    for i, e in events.iterrows():
        gfile, rfile = base + e["goes_fnames"], base + e["rad_fnames"]
        prfile = base + e["rad_fnames"].replace("data/rad_","proc/")
        resfname = rfile.replace("data/rad_", "i_results/ols_%s_"%key).replace(".csv", ".pickle")
        res = load_pickle(resfname)
        pred = (10**res.get_prediction().summary_frame(alpha=1-ci))-.1
        g = pd.read_csv(gfile, parse_dates=["time_tag"])
        u = pd.read_csv(rfile, parse_dates=["time"])
        fig = plt.figure(dpi=120, figsize=(5,6))
        ax = fig.add_subplot(211)
        ax.semilogy(g.time_tag, g.A_AVG, "b", label="HXR")
        ax.semilogy(g.time_tag, g.B_AVG, "r", label="SXR")
        ax.legend(loc=1)
        ax.set_ylim(1e-8,1e-2)
        ax.set_ylabel(r"$Watts.m^{-2}$")
        ax.set_xlim(u.time.tolist()[0], u.time.tolist()[-1])
        ax = fig.add_subplot(212)
        ux = pd.read_csv(prfile, parse_dates=["time"])
        ax.plot(ux.time, pred["mean"], "r", lw=1.2, label="Pred")
        ax.fill_between(ux.time, pred["obs_ci_lower"], pred["obs_ci_upper"], color="r", alpha=0.3, label="CI=50%")
        if key=="echoes": 
            ax.plot(u.time, u.echoes, "ko", ms=0.8, alpha=0.7, label="Obs")
            ax.set_ylabel(r"$\mathcal{E}_n$")
            ax.set_ylim(-1,30)
        if key=="absp":
            ax.plot(ux.time, ux["absp"], "ko", ms=0.8, alpha=0.7, label="Obs")
            ax.set_ylabel(r"$\frac{\bar{\mathcal{E}_n}}{\mathcal{E}_n}$")
            ax.set_ylim(-1,20)
        ax.legend(loc=1)
        ax.set_xlabel("Time, UT")
        ax.set_xlim(u.time.tolist()[0], u.time.tolist()[-1])
        fig.autofmt_xdate()
        fig.savefig(resfname.replace("i_results/","figures/").replace(".pickle", ".png"), bbox_inches="tight")
        plt.close()
    return

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

def maintain_event_list(base="../", event_list_files=["regressor/event_list.csv"], clean=False):
    date_list, radar_list, dur_list, gfnames = [], [], [], []
    for f in event_list_files:
        events = pd.read_csv(base + f, parse_dates=["date"])
        events = events[events.durations.notnull()]
        for _i, e in events.iterrows():
            dn, radars, dur = e["date"], e["radars"].split("-"), e["durations"]
            if len(radars)>0: gfnames.extend(["regressor/data/goes_%04d.csv"%_i]*len(radars))
            for r in radars:
                print("Event - ", dn, r, dur)
                date_list.append(dn)
                radar_list.append(r)
                dur_list.append(dur)
    x = pd.DataFrame()
    x["rad"], x["date"], x["durations"], x["rad_fnames"], x["goes_fnames"] = radar_list, date_list, dur_list,\
            ["regressor/data/rad_%04d.csv"%t for t in range(len(dur_list))], gfnames
    x.to_csv(base + "regressor/events.csv", header=True, index=False)

    for i, r in x.iterrows():
        if clean: os.system("rm -rf " + base+r["rad_fnames"] + " " + base+r["goes_fnames"])
        dates = [r["date"] - dt.timedelta(minutes=int(r["durations"]/2)),
                r["date"] + dt.timedelta(minutes=r["durations"])]
        rad = r["rad"]
        if not os.path.exists(base+r["goes_fnames"]):
            g = fetch_goes_data(dates, "g15")
            g.to_csv(base+r["goes_fnames"], header=True, index=False)
        if not os.path.exists(base+r["rad_fnames"]):
            u = box.get_med_filt_data_by_dates(rad, dates, low=10, high=90)
            u.to_csv(base+r["rad_fnames"], header=True, index=False)
    return

def fit_indp_ols(o, fname, key):
    f = FORMULAS[key].format(gamma=1, delta=2)
    model = smf.ols(formula=f, data=o)
    results = model.fit()
    results.save(fname)
    return

def fit_indp_wls(o, fname, key):
    f = FORMULAS[key].format(gamma=1, delta=2)
    model = smf.wls(formula=f, data=o,
            weights=1/(0.1+o.echoes))
    results = model.fit()
    results.save(fname)
    return

def create_fit_dataset(base="../", fname="regressor/events.csv"):
    events = pd.read_csv(base + fname, parse_dates=["date"])
    for i, e in events.iterrows():
        print("Event - ", e["date"], e["rad"], e["durations"], e["goes_fnames"], e["rad_fnames"])
        gfile, rfile = base + e["goes_fnames"], base + e["rad_fnames"]
        g = pd.read_csv(gfile, parse_dates=["time_tag"])
        g = g[(g.A_QUAL_FLAG==0) & (g.B_QUAL_FLAG==0)]
        g = g[["time_tag", "B_AVG", "A_AVG"]]
        g = g.rename(columns={"time_tag":"time"})
        g = g.set_index("time").resample("1s").interpolate()
        u = pd.read_csv(rfile, parse_dates=["time"])
        u.time = [t.replace(microsecond=0) for t in u.time]
        u = u.set_index("time").resample("1s").interpolate()
        u.beams, u.tfreq, u.echoes = np.rint(u.beams).astype(int), np.round(u.tfreq/1e3,1), np.rint(u.echoes).astype(int)
        g = g.iloc[60:len(u)+60]
        m = pd.merge(u,g, how="inner", left_index=True, right_index=True).reset_index()
        m["absp"] = np.median(m.iloc[:180].echoes)/(m.echoes+1)
        fit_indp_ols(m, rfile.replace("data/rad_", "i_results/ols_echoes_").replace(".csv", ".pickle"), "echoes")
        fit_indp_ols(m, rfile.replace("data/rad_", "i_results/ols_absp_").replace(".csv", ".pickle"), "absp")
        fit_indp_wls(m, rfile.replace("data/rad_", "i_results/wls_echoes_").replace(".csv", ".pickle"), "echoes")
        fit_indp_wls(m, rfile.replace("data/rad_", "i_results/wls_absp_").replace(".csv", ".pickle"), "absp")
        m.to_csv(rfile.replace("data/rad_", "proc/"), header=True, index=False)
    generate_indp_fitted_plots("echoes")
    generate_indp_fitted_plots("absp")
    return

if __name__ == "__main__":
    #maintain_event_list()
    #create_fit_dataset()
    #fit_global_absorption_models()
    create_global_distribution([dt.datetime(2015,5,5,22), dt.datetime(2015,5,5,22,11)])
    pass
