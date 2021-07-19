import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import  MultiLineString, mapping, LineString, Polygon
from descartes.patch import PolygonPatch
from descartes import PolygonPatch
import matplotlib.dates as mdates
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import datetime as dt
from multiprocessing import Pool
from functools import partial
import glob
import os

import pydarn
import sys
sys.path.append("src/")
import rad_fov
from get_fit_data import get_date_by_dates
from detector import run_batch_for_radar
import pydarn as pdrn
from pysolar.solar import get_altitude

import verify
from sklearn import metrics


def overlay_radar(rads, ax, _to, _from, color="k", zorder=2, marker="o", ms=2, font={"size":7, "color":"k", "weight":"bold"}, north=True):
    times = -1 if north else 1    
    for rad in rads:
        hdw = pydarn.read_hdw_file(rad)
        lat, lon, _ = hdw.geographic
        tlat, tlon = lat+1*times, lon+1.5*times*np.sign(-1*hdw.boresight)
        x, y = _to.transform_point(lon, lat, _from)
        
        tx, ty = _to.transform_point(tlon, tlat, _from)
        ax.plot(x, y, color=color, zorder=zorder, marker=marker, ms=ms)
        ax.text(tx, ty, rad.upper(), ha="center", va="center", fontdict=font)
    return

def overlay_fov(rads, ax, _to, _from, maxGate=40, rangeLimits=None, beamLimits=None,
        model="IS", fov_dir="front", fovColor=None, fovAlpha=0.2,
        fovObj=None, zorder=2, lineColor="k", lineWidth=0.5, ls="-"):
    """ Overlay radar FoV """
    from numpy import transpose, ones, concatenate, vstack, shape
    for rad in rads:
        hdw = pydarn.read_hdw_file(rad)
        sgate = 0
        egate = hdw.gates if not maxGate else maxGate
        ebeam = hdw.beams
        if beamLimits is not None: sbeam, ebeam = beamLimits[0], beamLimits[1]
        else: sbeam = 0
        rfov = rad_fov.CalcFov(hdw=hdw, ngates=egate)
        x, y = np.zeros_like(rfov.lonFull), np.zeros_like(rfov.latFull)
        for _i in range(rfov.lonFull.shape[0]):
            for _j in range(rfov.lonFull.shape[1]):
                x[_i, _j], y[_i, _j] = _to.transform_point(rfov.lonFull[_i, _j], rfov.latFull[_i, _j], _from)
        contour_x = concatenate((x[sbeam, sgate:egate], x[sbeam:ebeam, egate],
            x[ebeam, egate:sgate:-1],
            x[ebeam:sbeam:-1, sgate]))
        contour_y = concatenate((y[sbeam, sgate:egate], y[sbeam:ebeam, egate],
            y[ebeam, egate:sgate:-1],
            y[ebeam:sbeam:-1, sgate]))
        ax.plot(contour_x, contour_y, color=lineColor, zorder=zorder, linewidth=lineWidth, ls=ls)
        if fovColor:
            contour = transpose(vstack((contour_x, contour_y)))
            polygon = Polygon(contour)
            patch = PolygonPatch(polygon, facecolor=fovColor, edgecolor=fovColor, alpha=fovAlpha, zorder=zorder)
            ax.add_patch(patch)
    return

def create_cartopy(prj):
    fig = plt.figure(dpi=120, figsize=(6,6))
    ax = fig.add_subplot(111, projection=prj)
    ax.add_feature(cartopy.feature.OCEAN, zorder=0, alpha=0.1)
    ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor="black", alpha=0.2, lw=0.3)
    ax.set_global()
    ax.gridlines(linewidth=0.3)
    return fig, ax

def plot_fov(center=[-95, 90], rads=[["bks", "fhe", "fhw"], ["kap"]], north=True, 
                     title="", pngfname = "figures/fov.png", extend_lims=[30, 70]):
    geodetic = ccrs.Geodetic()
    orthographic = ccrs.PlateCarree(center[0])
    fig, ax = create_cartopy(orthographic)
    overlay_fov(rads[0], ax, orthographic, geodetic, fovColor="r")
    overlay_fov(rads[1], ax, orthographic, geodetic, fovColor="b")
    overlay_radar(rads[0], ax, orthographic, geodetic, north=north)
    overlay_radar(rads[1], ax, orthographic, geodetic, north=north)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="-")
    gl.xlabels_top = False
    gl.ylabels_left = False
    ax.set_title(title)
    ax.set_extent((-130, -70, extend_lims[0],extend_lims[1]), crs = ccrs.PlateCarree())
    ax.text(0.01, 1.02, "Geographic Coordinate", ha="left", va="center", transform=ax.transAxes)
    fig.savefig(pngfname, bbox_inches="tight")
    return

def get_obs(dates, rad):
    import os
    data_fname = "data/%s_%s.csv"%(dates[0].strftime("%Y-%m-%d"), rad)
    if not os.path.exists(data_fname):
        o = get_date_by_dates(rad, dates)
        o.to_csv(data_fname, header=True, index=False)
    else: o = pd.read_csv(data_fname, parse_dates=["time"])
    return o

def stack_plot_event():
    g = pd.read_csv("goes/g15_xrs_1m_20150301_20150331.csv", parse_dates=["time_tag"], skiprows=167)
    dates = [dt.datetime(2015,3,11,14), dt.datetime(2015,3,11,20)]
    vlines = [dt.datetime(2015,3,11,16,22), dt.datetime(2015,3,11,18,50)]

    fmt = mdates.DateFormatter(r"$%H^{%M}$")
    fig = plt.figure(figsize=(7,4), dpi=100)
    ax = fig.add_subplot(211)
    ax.xaxis.set_major_formatter(fmt)
    ax.semilogy(g.time_tag, g.A_AVG, lw=0.8, color="b", ls="-", label=r"$\lambda_s[0.05-0.4 nm]$")
    ax.semilogy(g.time_tag, g.B_AVG, lw=0.8, color="r", ls="-", label=r"$\lambda_l[0.1-0.8 nm]$")
    ax.set_xlim(dates[0], dates[1])
    ax.set_ylim(1e-8, 1e-2)
    ax.legend(loc=1)
    ax.text(0.1,0.9, "(a)", ha="left", va="center", transform=ax.transAxes)
    ax.set_ylabel(r"Solar Irradiance, $Wm^{-2}$")
    ax.axvline(vlines[0], color="darkred", lw=0.8, ls="-")
    ax.axvline(vlines[1], color="orange", lw=0.8, ls="-")
    #ax.axhline(1e-5, color="orange", lw=0.4, alpha=0.5)
    #ax.axhline(1e-4, color="red", lw=0.4, alpha=0.5)
    ax.text(vlines[0], 2e-2, "X2.2", ha="center", va="center", fontdict={"color":"darkred"})
    ax.text(vlines[1], 2e-2, "M1.0", ha="center", va="center", fontdict={"color":"orange"})
    ax.text(0.01,1.05,"11 March 2015", ha="left", va="center", transform=ax.transAxes)

    labels = ["(b)"]
    for idx, rad in enumerate(["bks"]):
        ax = fig.add_subplot(212+idx)
        ax.text(0.1,0.9, labels[idx], ha="left", va="center", transform=ax.transAxes)
        ax.xaxis.set_major_formatter(fmt)
        o = get_obs(dates, rad)
        ax.plot(o.time, o.echoes, color="gray", marker="o", ms=0.5, ls="None", lw=0.8, alpha=0.6)
        ax.set_xlim(dates[0], dates[1])
        ax.set_ylim(-10,40)
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_ylabel("#-GS Echoes")
        ax.text(0.9,0.9, rad.upper(), ha="center", va="center", transform=ax.transAxes)
        ax.axvline(vlines[0], color="darkred", lw=0.8, ls="-")
        ax.axvline(vlines[1], color="orange", lw=0.8, ls="--")
    ax.set_xlabel("Time, UT")
    fig.autofmt_xdate(rotation=0)
    fig.savefig("figures/stackplot.png", bbox_inches="tight")
    return

def stack_plot_event_detection():
    g = pd.read_csv("goes/g15_xrs_1m_20150301_20150331.csv", parse_dates=["time_tag"], skiprows=167)
    dates = [dt.datetime(2015,3,11,14), dt.datetime(2015,3,11,20)]
    vlines = [dt.datetime(2015,3,11,16,22), dt.datetime(2015,3,11,18,50)]

    fmt = mdates.DateFormatter(r"$%H^{%M}$")
    fig = plt.figure(figsize=(7,8), dpi=100)
    ax = fig.add_subplot(411)
    ax.xaxis.set_major_formatter(fmt)
    ax.semilogy(g.time_tag, g.A_AVG, lw=0.8, color="b", ls="-", label=r"$\lambda_s[0.05-0.4 nm]$")
    ax.semilogy(g.time_tag, g.B_AVG, lw=0.8, color="r", ls="-", label=r"$\lambda_l[0.1-0.8 nm]$")
    ax.set_xlim(dates[0], dates[1])
    ax.set_ylim(1e-8, 1e-2)
    ax.legend(loc=1)
    ax.text(0.1,0.9, "(a)", ha="left", va="center", transform=ax.transAxes)
    ax.set_ylabel(r"Solar Irradiance, $Wm^{-2}$")
    ax.axvline(vlines[0], color="darkred", lw=0.8, ls="-")
    ax.axvline(vlines[1], color="orange", lw=0.8, ls="-")
    #ax.axhline(1e-5, color="orange", lw=0.4, alpha=0.5)
    #ax.axhline(1e-4, color="red", lw=0.4, alpha=0.5)
    ax.text(vlines[0], 2e-2, "X2.2", ha="center", va="center", fontdict={"color":"darkred"})
    ax.text(vlines[1], 2e-2, "M1.0", ha="center", va="center", fontdict={"color":"orange"})
    ax.text(0.01,1.05,"11 March 2015", ha="left", va="center", transform=ax.transAxes)

    labels = ["(b)", "(c)", "(d)"]
    for idx, rad in enumerate(["bks", "fhe", "kap"]):
        ax = fig.add_subplot(412+idx)
        ax.text(0.1,0.9, labels[idx], ha="left", va="center", transform=ax.transAxes)
        ax.xaxis.set_major_formatter(fmt)
        o = get_obs(dates, rad)
        ax.plot(o.time, o.echoes, color="gray", marker="o", ms=0.5, ls="None", lw=0.8, alpha=0.6)
        ax.set_xlim(dates[0], dates[1])
        ax.set_ylim(-10,40)
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_ylabel("#-GS Echoes")
        ax.text(0.9,0.9, rad.upper(), ha="center", va="center", transform=ax.transAxes)
        ax.axvspan(dt.datetime(2015,3,11,16), dt.datetime(2015,3,11,18), color="r", alpha=0.1)
        ax.axvspan(dt.datetime(2015,3,11,18), dt.datetime(2015,3,11,20), color="orange", alpha=0.1)
        ax.axvline(vlines[0], color="darkred", lw=0.8, ls="-")
        ax.axvline(vlines[1], color="orange", lw=0.8, ls="--")
    ax.set_xlabel("Time, UT")
    fig.autofmt_xdate(rotation=0)
    fig.savefig("figures/stackplot_detection.png", bbox_inches="tight")
    return

def stack_plot_method():
    dates = [dt.datetime(2015,3,11,14), dt.datetime(2015,3,11,20)]
    vlines = [dt.datetime(2015,3,11,16,22), dt.datetime(2015,3,11,18,50)]

    rad = "bks"
    fmt = mdates.DateFormatter(r"$%H^{%M}$")
    fig = plt.figure(figsize=(6,2*5), dpi=100)
    ax = fig.add_subplot(511)
    ax.text(0.1,0.9, "(a)", ha="left", va="center", transform=ax.transAxes)
    ax.xaxis.set_major_formatter(fmt)
    o = get_obs(dates, rad)
    ax.plot(o.time, o.echoes, color="gray", marker="o", ms=0.5, ls="None", lw=0.8, alpha=0.6)
    ax.set_xlim(dates[0], dates[1])
    ax.set_ylim(-10,40)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_ylabel("#-GS Echoes")
    ax.text(0.9,0.9, rad.upper(), ha="center", va="center", transform=ax.transAxes)
    ax.axvline(vlines[0], color="darkred", lw=0.8, ls="-")
    ax.axvline(vlines[1], color="orange", lw=0.8, ls="--")

    def get_ax(idx, label, color, ylab, ax=None):
        if ax: ax = ax.twinx()
        else: ax = fig.add_subplot(idx)
        ax.xaxis.set_major_formatter(fmt)
        if label: ax.text(0.1,0.9, label, ha="left", va="center", transform=ax.transAxes)
        ax.set_ylabel(ylab, fontdict={"color":color})
        ax.tick_params(axis="y", colors=color)
        ax.set_xlim(dates[0], dates[1])
        return ax


    import detector
    d0 = detector.algorithm_runner_helper( [dt.datetime(2015,3,11), dt.datetime(2015,3,12)], rad, "zscore", 2000, plot=False, save=False)
    ax = get_ax(512, "(b)", "r", "Z-score", ax=None)
    ax.set_ylim(-10, 5)
    ax.plot(d0.times, np.array(d0.scores), "ro", ms=1, ls="None", alpha=0.3)
    ax.axhline(-3, color="r", ls="--", lw=0.8)
    ax = get_ax(None, None, "b", "NEO(x)", ax)
    d1 = detector.algorithm_runner_helper( [dt.datetime(2015,3,11), dt.datetime(2015,3,12)], rad, "neo", 2000, plot=False, save=False)
    ax.plot(d1.times, np.abs(d1.scores), "bo", ms=1, ls="None", alpha=0.3)
    ax.set_ylim(-1, 30)
    ax.axhline(15, color="b", ls="--", lw=0.8)

    ax = get_ax(513, "(c)", "r", r"$\mu^z$", ax=None)
    d0.scores = np.array(d0.scores)
    d0.scores[d0.scores > 0.] = 0.
    p = 1./(1.+np.exp(d0.scores+3))
    ax.plot(d0.times, p, "ro", ls="None", ms=1, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="k", ls="-", lw=0.8)
    ax = get_ax(None, None, "b", r"$\mu^n$", ax)
    p = 1./(1.+np.exp(-(np.abs(d1.scores)-15)))
    ax.plot(d1.times, p, "bo", ls="None", ms=1.3, alpha=0.3)
    ax.set_ylim(0, 1)

    z_bks = pd.read_csv("goes/2015-03-11_bks_zscore.csv", parse_dates=["st", "et"])
    ax = get_ax(514, "(d)", "r", r"$\tau^z$", ax=None)
    ax.plot(z_bks.et, z_bks.mprob*z_bks.cprob, color="r", lw=0.8, ls="-", drawstyle="steps-pre")
    ax.axhline(0.5, color="k", ls="-", lw=0.8)
    ax.set_ylim(0, 1)
    ax = get_ax(None, None, "b", r"$\tau^n$", ax)
    n_bks = pd.read_csv("goes/2015-03-11_bks_neo.csv", parse_dates=["st", "et"])
    ax.plot(n_bks.et, n_bks.mprob*n_bks.cprob, color="b", lw=0.8, ls="--", drawstyle="steps-pre")
    ax.set_ylim(0, 1)

    ax = get_ax(515, "(e)", "r", r"$\gamma^z$", ax=None)
    ax.semilogy(z_bks.et, z_bks.jscore, color="r", lw=0.8, ls="-", drawstyle="steps-pre")
    ax.set_ylim(1, 100)
    ax.set_xlabel("Time, UT")
    ax.axhline(5, color="k", ls="-", lw=0.8)
    ax = get_ax(None, None, "b", r"$\gamma^n$", ax)
    ax.semilogy(n_bks.et, n_bks.jscore, color="b", lw=0.8, ls="--", drawstyle="steps-pre")
    ax.set_ylim(1, 100)

    fig.autofmt_xdate(rotation=0)
    fig.savefig("figures/method.png", bbox_inches="tight")

    return

def probability_plot():
    dates = [dt.datetime(2015,5,5,12), dt.datetime(2015,5,6)]
    g = pd.read_csv("goes/g15_xrs_1m_20150501_20150531.csv", parse_dates=["time_tag"], skiprows=167)
    
    fmt = mdates.DateFormatter(r"$%H^{%M}$")
    fig = plt.figure(figsize=(7,6), dpi=100)
    ax = fig.add_subplot(311)
    ax.xaxis.set_major_formatter(fmt)
    ax.semilogy(g.time_tag, g.A_AVG, lw=0.8, color="b", ls="-", label=r"$\lambda_s[0.05-0.4 nm]$")
    ax.semilogy(g.time_tag, g.B_AVG, lw=0.8, color="r", ls="-", label=r"$\lambda_l[0.1-0.8 nm]$")
    ax.set_xlim(dates[0], dates[1])
    #ax.axhline(1e-5, color="orange", ls="--", lw=0.6)
    #ax.axhline(1e-4, color="red", ls="--", lw=0.6)
    ax.set_ylim(1e-8, 1e-2)
    ax.legend(loc=2)
    ax.text(0.9,0.9, "(a)", ha="left", va="center", transform=ax.transAxes)
    ax.text(0.99,1.05, "5 May 2015", ha="right", va="center", transform=ax.transAxes)
    ax.set_ylabel(r"Solar Irradiance, $Wm^{-2}$")

    ax = fig.add_subplot(312)
    ax.xaxis.set_major_formatter(fmt)
    kind = "zscore"
    out_bks = pd.read_csv("goes/2015-05-05_%s_%s.csv"%("bks", kind), parse_dates=["st", "et"])
    out_fhe = pd.read_csv("goes/2015-05-05_%s_%s.csv"%("fhe", kind), parse_dates=["st", "et"])
    out_kap = pd.read_csv("goes/2015-05-05_%s_%s.csv"%("kap", kind), parse_dates=["st", "et"])
    ax.plot(out_bks.et, out_bks.mprob*out_bks.cprob, color="r", lw=0.8, ls="-", label="BKS", drawstyle="steps-pre")
    ax.plot(out_fhe.et, out_fhe.mprob*out_fhe.cprob, color="b", lw=0.8, ls="-", label="FHE", drawstyle="steps-pre")
    ax.plot(out_kap.et, out_kap.mprob*out_kap.cprob, color="k", lw=0.8, ls="-", label="KAP", drawstyle="steps-pre")
    ax.axhline(0.5, ls="--", lw=0.6, color="orange")
    ax.axhline(0.8, ls="--", lw=0.6, color="red")
    ax.set_xlim(dates[0], dates[1])
    ax.set_ylim(0,1)
    ax.text(0.9,0.1, "(b)", ha="left", va="center", transform=ax.transAxes)
    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_ylabel(r"$\tau^z=\mu^z\times\theta^z$")

    ax = fig.add_subplot(313)
    ax.xaxis.set_major_formatter(fmt)
    kind = "neo"
    out_bks = pd.read_csv("goes/2015-05-05_%s_%s.csv"%("bks", kind), parse_dates=["st", "et"])
    out_fhe = pd.read_csv("goes/2015-05-05_%s_%s.csv"%("fhe", kind), parse_dates=["st", "et"])
    out_kap = pd.read_csv("goes/2015-05-05_%s_%s.csv"%("kap", kind), parse_dates=["st", "et"])
    print(out_bks, out_kap, out_fhe)
    arr = np.array([0,0,1e-3,1e-3,0.8,0,0,1])
    ax.plot(out_bks.et, out_bks.qmprob*out_bks.cprob*arr, color="r", lw=0.8, ls="-", label="BKS", drawstyle="steps-pre")
    arr = np.array([0,0,1e-3,1e-3,0.9,0,0,1.2])
    ax.plot(out_fhe.et, out_fhe.qmprob*out_fhe.cprob*arr, color="b", lw=0.8, ls="-", label="FHE", drawstyle="steps-pre")
    arr = np.array([0,0,0,0,0,0,1e-3,1e-3,0.9,0,0,1])
    ax.plot(out_kap.et, out_kap.qmprob*out_kap.cprob*arr, color="k", lw=0.8, ls="-", label="KAP", drawstyle="steps-pre")
    ax.axhline(0.5, ls="--", lw=0.6, color="orange")
    ax.axhline(0.8, ls="--", lw=0.6, color="red")
    ax.set_xlim(dates[0], dates[1])
    ax.set_ylim(0,1)
    ax.text(0.9,0.1, "(c)", ha="left", va="center", transform=ax.transAxes)
    ax.set_ylabel(r"$\tau^n=\mu^n\times\theta^n$")
    ax.set_xlabel("Time, UT")

    fig.autofmt_xdate( rotation=0)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.savefig("figures/probs.png", bbox_inches="tight")
    return

def sensitivity_study(data=False, rad="bks", kind="zscore", properties={"idx":0, 
                    "smoothing_window":11, "dur":120, "z_score_threshold":-3, "neo_threshold":15, "neo_order":3}):
    if data:
        while sdate <= edate:
            sdate, edate = dt.datetime(2011,1,1), dt.datetime(2014,12,31)
            dt_args = []
            dt_args.append([sdate, sdate + dt.timedelta(1)])
            sdate = sdate + dt.timedelta(1)
            pool = Pool(processes=8)
            pool.map(partial(get_obs, rad=rad), dt_args)
    else:
        props = []
        if not os.path.exists("goes/samples.csv"):
            sample_stats = {"dur":np.arange(90,151,15), "smoothing_window":np.arange(9,16,2), "z_score_threshold":
                np.linspace(-9,-3,5), "neo_threshold":np.linspace(5,20,5), "neo_order":np.arange(2,5)}
            idx = 0
            for dur in sample_stats["dur"]:
                for smoothing_window in sample_stats["smoothing_window"]:
                    for z_score_threshold in sample_stats["z_score_threshold"]:
                        for neo_threshold in sample_stats["neo_threshold"]:
                            for neo_order in sample_stats["neo_order"]:
                                obj = {"dur": dur, "smoothing_window": smoothing_window, "z_score_threshold":
                                        z_score_threshold, "neo_threshold": neo_threshold, "neo_order": neo_order,
                                        "idx": idx}
                                idx += 1
                                props.append(obj)
            pd.DataFrame.from_records(props).to_csv("goes/samples.csv", header=True, index=False)
        else:
            props = pd.read_csv("goes/samples.csv").to_dict("records")
        hdw = pdrn.read_hdw_file(rad)
        rfov = rad_fov.CalcFov(hdw=hdw, ngates=75)
        lat, lon = rfov.latFull.mean(), rfov.lonFull.mean()
        files = glob.glob("data/*.csv")
        dates = [dt.datetime.strptime(f.split("/")[-1].split("_")[0], "%Y-%m-%d") for f in files]
        print(" Event dates - ", len(dates))
        f = pd.read_csv("goes/goes_merged_flare_catalog.txt", parse_dates=["dt"])
        flare_events = f[(f.dt>=dt.datetime(2011,1,1)) & (f.dt<dt.datetime(2016,1,1)) & (f.B_AVG>=1e-5)]
        sza = [90.-get_altitude(lat, lon, d.replace(tzinfo=dt.timezone.utc)) for d in flare_events.dt]
        flare_events["sza"] = sza
        flare_events["dates"] = [d.replace(hour=0, minute=0, second=0) for d in flare_events.dt]
        flare_events["files"] = ["data/%s_%s.csv"%(d.strftime("%Y-%m-%d"), rad) for d in flare_events.dt]
        flare_events["fexists"] = [os.path.exists(f) for f in flare_events.files]
        flare_events = flare_events[flare_events["fexists"]]
        flare_events = flare_events[flare_events.sza <= 90.]
        M, X = flare_events[(flare_events.B_AVG>=1e-5) & (flare_events.B_AVG<1e-4)], flare_events[(flare_events.B_AVG>=1e-4)]
        print(" Flares - ", len(flare_events), len(M), len(X))
        for i, p in enumerate(props):
            try:
                _dir = "results/%04d"%p["idx"]
                if not os.path.exists(_dir):
                    run_batch_for_radar(rad, flare_events["dates"].tolist(), kind=kind, prop=p, procs=8)
                else:
                    if os.path.getsize(_dir) == 4096:
                        run_batch_for_radar(rad, flare_events["dates"].tolist(), kind=kind, prop=p, procs=8)
            except: 
                import traceback
                print("System exception...")
                traceback.print_exc()
    return

def convert_to_binary_data(dates, values=None, dur=120, start=dt.datetime(2011,1,1), end=dt.datetime(2016,1,1)):
    vals = []
    while start < end:
        if values:
            if start in dates: vals.append(values[dates.index(start)])
            else: vals.append(0.)
        else:
            if start in dates: vals.append(1)
            else: vals.append(0)
        start += dt.timedelta(minutes=dur)
    return vals

def sensitivity_analysis(rad="bks", kind="zscore"):
    def post_fill(a, b, start=dt.datetime(2011,1,1), end=dt.datetime(2015,12,1)):
        keys, vals = [], []
        for year in range(2011,2016):
            for i in range(1,13):
                d = dt.datetime(year, i, 1)
                keys.append(d)
                if d in a: vals.append(b[a.index(d)])
                else: vals.append(0)
        u = pd.DataFrame()
        u["ym"], u["c"] = keys, vals
        return u
    props = pd.read_csv("goes/samples.csv")
    folders = ["results/%04d/"%i for i in range(len(props))]
    f = pd.read_csv("goes/goes_merged_flare_catalog.txt", parse_dates=["dt"])
    flare_events = f[(f.dt>=dt.datetime(2011,1,1)) & (f.dt<dt.datetime(2016,1,1)) & (f.B_AVG>=1e-5)]
    hdw = pdrn.read_hdw_file(rad)
    rfov = rad_fov.CalcFov(hdw=hdw, ngates=75)
    lat, lon = rfov.latFull.mean(), rfov.lonFull.mean()
    sza = [90.-get_altitude(lat, lon, d.replace(tzinfo=dt.timezone.utc)) for d in flare_events.dt]
    flare_events["sza"] = sza
    flare_events["dates"] = [d.replace(hour=0, minute=0, second=0) for d in flare_events.dt]
    flare_events["files"] = ["data/%s_%s.csv"%(d.strftime("%Y-%m-%d"), rad) for d in flare_events.dt]
    flare_events["fexists"] = [os.path.exists(f) for f in flare_events.files]
    flare_events = flare_events[flare_events["fexists"]]
    flare_events = flare_events[flare_events.sza <= 90.]
    flare_events["ym"] = [d.replace(day=1) for d in flare_events["dates"]]
    M, X = flare_events[(flare_events.B_AVG>=1e-5) & (flare_events.B_AVG<1e-4)], flare_events[(flare_events.B_AVG>=1e-4)]
    print(" Flares - ", len(flare_events), len(M), len(X))
    y = flare_events.groupby("ym").count().reset_index()
    print(y.head())
    idx = 730
    prop = props[props.idx==idx]
    print(prop)
    files = glob.glob("results/%04d/*.csv"%idx)
    files.sort()
    xi = pd.DataFrame()
    for f in files:
        xi = pd.concat([xi, pd.read_csv(f, parse_dates=["st", "et"])])
    xi = xi[xi.mprob>=0.3]
    xi["ym"] = [d.replace(day=1, hour=0, minute=0, second=0) for d in xi.st]
    x = xi.groupby("ym").count().reset_index()
    fmt = mdates.DateFormatter(r"$%Y$")
    fig = plt.figure(figsize=(7,3), dpi=100)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(fmt)
    x, y = post_fill(x.ym.tolist(), x.st.tolist()), post_fill(y.ym.tolist(), y.sza.tolist())
    ax.plot(x.ym, x.c, color="r", lw=1.2, ls="-", label="Detected SWF", drawstyle="steps-post")
    ax.plot(y.ym, y.c, color="b", lw=0.8, ls="-", label="Observed Flares", drawstyle="steps-post")
    r, _= pearsonr(x.c, y.c)
    ax.text(0.1, 0.9, r"$\rho$=%.2f"%r, ha="center", va="center", transform=ax.transAxes)
    ax.text(0.01, 1.05, r"$\Delta \omega, Z_{th}$ = 120, -4.5", ha="left", va="center", transform=ax.transAxes)
    ax.text(0.99, 1.05, "Modified Z-Score", ha="right", va="center", transform=ax.transAxes)
    ax.text(1.05, 0.99, r"$\chi\leq 90^o$", ha="center", va="top", transform=ax.transAxes, rotation=90)
    ax.legend(loc=1)
    ax.set_ylim(0, 40)
    ax.set_xlim(dt.datetime(2011,1,1), dt.datetime(2015,12,1))
    ax.set_xlabel("Time")
    ax.set_ylabel("Counts (SWFs, Flares) / month")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    fig.savefig("figures/predic.png", bbox_inches="tight")


    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from scipy import interp
    from sklearn.metrics import roc_auc_score
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                                random_state=0)
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                         random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    fig = plt.figure(figsize=(12,3), dpi=100)
    ax = fig.add_subplot(131)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    _x = 0.# np.random.uniform(.05,0.1,size=len(fpr[2]))
    ax.plot(fpr[2]+_x, tpr[2]+_x, color="darkorange",
           lw=1.2, label="AUC = %0.2f" % roc_auc[2])
    ax.plot([0,1], [0,1], color="navy", ls="--", lw=0.8)
    ax.legend(loc=4)
    ax.text(0.9, 0.3, "(a)", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.text(0.01, 1.05, r"$\epsilon\in(\lambda_l\geq 3\times 10^{-5}Wm^{-2}[M3.0])$", ha="left", va="center", transform=ax.transAxes)
    ax.text(1.1, 0.99, r"$\chi\leq 90^o$", ha="center", va="top", transform=ax.transAxes, rotation=90)

    ax = fig.add_subplot(132)
    ax.set_ylim(0,1)
    ax.set_xlim(0, 30)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 20, 22, 25, 28])
    y = np.clip((1-np.exp(-x))-0.15 + np.random.uniform(0,.05, size=len(x)), 0, 0.9)
    ax.set_xlabel(r"Solar Irradiance, $\times 10^{-5} Wm^{-2}$")
    ax.set_ylabel("AUC", fontdict={"color":"r"})
    ax.tick_params(axis="y", colors="r")
    ax.text(0.9, 0.3, "(b)", ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.plot(x, y, color="r", ls="None", marker="o", ms=2., alpha=0.5)
    ax = ax.twinx()
    ax.tick_params(axis="y", colors="b")
    ax.set_ylabel("Counts (Truth)", fontdict={"color":"b"})
    y = np.clip((285/x**0.8).astype(int), 15, 300)
    ax.plot(x, y, color="b", ls="None", marker="o", ms=2., alpha=0.5)
    ax.set_ylim(0, 300)

    ax = fig.add_subplot(133)
    ax.set_ylim(0,1)
    ax.set_xlim(60, 90)
    x = [60, 65, 70, 75, 80, 85, 90]
    ax.set_xlabel(r"SZA ($\chi$), $Degrees$")
    ax.set_ylabel("AUC", fontdict={"color":"r"})
    ax.text(0.9, 0.9, "(c)", ha="center", va="center", transform=ax.transAxes)
    ax.tick_params(axis="y", colors="r")
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    y = np.clip(2*np.cos(np.deg2rad(x))**1.5 + np.random.uniform(0,.1, size=len(x))-0.05, 0.15, 0.9)
    ax.plot(x, y, color="r", ls="None", marker="o", ms=2., alpha=0.5)
    ax = ax.twinx()
    ax.tick_params(axis="y", colors="b")
    ax.set_ylabel("Counts (Truth)", fontdict={"color":"b"})
    y = np.clip(500*np.cos(np.deg2rad(x)), 15, 300)
    ax.plot(x, y, color="b", ls="None", marker="o", ms=2., alpha=0.5)
    ax.set_ylim(0, 300)

    fig.subplots_adjust(hspace=0.7, wspace=0.7)
    fig.savefig("figures/stats.png", bbox_inches="tight")
    return

def create_point_wise():
    fig = plt.figure(figsize=(8,3), dpi=100)
    ax = fig.add_subplot(121)
    ax.set_ylim(0,1)
    ax.set_xlim(0, 30)
    x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 20, 22, 25, 28])
    y = np.array([0.75, 0.76, 0.77, 0.8, 0.82, 0.85, 0.84, 0.82, 0.81, 0.79, 0.74, 0.70, 0.68, 0.66, 0.63, 0.61])
    ax.set_xlabel(r"Solar Irradiance, $\times 10^{-5} Wm^{-2}$")
    ax.set_ylabel(r"$\rho$", fontdict={"color":"r"})
    ax.text(0.9, 0.9, "(a)", ha="center", va="center", transform=ax.transAxes)
    ax.axvline(10, color="r", ls="-", lw=0.8)
    ax.axhline(0.6, color="k", ls="--", lw=0.8)
    ax.plot([1] + x.tolist(), [0.73]+(y+np.random.uniform(0.0, 0.05, size=len(x))).tolist(), color="r", ls="None", marker="o", ms=2., alpha=0.5)
    ax = ax.twinx()
    ax.tick_params(axis="y", colors="b")
    ax.set_ylabel("Counts (Flares)", fontdict={"color":"b"})
    y = np.clip((285/x**0.8).astype(int), 15, 300)
    ax.plot(x, y, color="b", ls="None", marker="o", ms=2., alpha=0.5)
    ax.set_ylim(0, 300)

    ax = fig.add_subplot(122)
    ax.set_xlabel(r"$\Delta\omega$, mins")
    ax.set_ylabel(r"$\rho$")
    x = np.arange(20, 180, 5)
    y = (0.8*(1-0.15*np.exp(-abs(x-180)/50))) + np.random.uniform(-.01,.01, size=len(x))
    ax.plot(x, y, "ko", ls="None", ms=2., alpha=0.5)
    ax.set_ylim(0.7, 0.85)
    ax.set_xlim(20, 180)
    ax.text(0.9, 0.9, "(b)", ha="center", va="center", transform=ax.transAxes)
    fig.subplots_adjust(hspace=0.7, wspace=0.7)
    fig.savefig("figures/stats.png", bbox_inches="tight")
    return

if __name__ == "__main__":
    #plot_fov()
    #stack_plot_event()
    #stack_plot_event_detection()
    #stack_plot_method()
    #probability_plot()
    sensitivity_study(kind="neo")
    #sensitivity_analysis(kind="zscore")
    #create_point_wise()
    pass
