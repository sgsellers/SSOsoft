import glob
import os
import sys

import astropy.io.fits as fits
import matplotlib.animation as animation
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a

import ssosoft.spectral.spectraTools as spectraTools


def ibis_movie_maker(
        ibis_filelist: list, save_directory: str,
        fps: int=5, progress: bool=True
    ) -> None:
    """
    Makes movie of IBIS NB data. Processes all filters in a given directory

    Parameters
    ----------
    ibis_filelist : List of Level-1 NB IBIS data
    save_directory : directory to save movies to
    fps : int, optional
        FPS of final movie, default 5
    progress : bool, optional
        Whether to spawn tqdm bar, default True

    """
    directory = os.path.split(ibis_filelist[0])
    filts = list(set([os.path.split(i)[1].split("_")[0] for i in ibis_filelist if os.path.split(i)[1].split("_")[0].isnumeric()]))
    for filt in filts:
        flist = sorted(glob.glob(os.path.join(directory, filt + "*.fits")))
        with fits.open(flist[0]) as hdul:
            starttime = np.datetime64(hdul[0].header["DATE"]) + np.timedelta64(int(1000 * hdul[0].header["TIME"]), "ms")
        with fits.open(flist[-1]) as hdul:
            endtime = np.datetime64(hdul[-1].header["DATE"]) + np.timedelta64(int(1000 * hdul[-1].header["TIME"]), "ms")
        date = starttime.astype("datetime64[D]").astype(str).replace("-", "")
        time = starttime.astype(str).split("T")[1].replace(":", "").split(".")[0]
        # Check to see if movie exists and skip loop if needed
        outfile = os.path.join(save_directory, "ibis_" + filt + "_" + date + "_" + time + ".mp4")
        if os.path.exists(outfile):
            continue
        with fits.open(flist[0]) as hdul:
            waves = np.asarray([i.header["WAVE"] for i in hdul])
            grid = np.asarray([i.header["GRID"] for i in hdul])
        goes_timestamps, short, long = get_goes_timeseries(starttime, endtime, save_directory)
        fts_w, fts_s = spectraTools.fts_window(waves.min() - 0.5, waves.max() + 0.5)
        # Set up lightcurve, scintillation
        scin = []
        llvl = []
        ibis_ts = []
        lc_ts = []
        lc = []
        for file in flist:
            with fits.open(file) as hdul:
                if file == flist[0]:
                    img = hdul[0].data
                core_im = hdul[np.argmin(np.abs(grid))].data
                core_hdr = hdul[np.argmin(np.abs(grid))].header
                lc.append(np.mean(core_im[core_im != core_im[0, 0]]))
                lc_ts.append(np.datetime64(core_hdr["DATE"]) + np.timedelta64(int(1000 * core_hdr["TIME"]), "ms"))
                for hdu in hdul:
                    scin.append(hdu.header["DST_SEE"])
                    llvl.append(hdu.header["DST_LLEV"])
                    ibis_ts.append(
                        np.datetime64(hdu.header["DATE"]) + np.timedelta64(int(1000 * hdu.header["TIME"]), "ms"))
        vmin = []
        vmax = []
        with fits.open(flist[int(len(flist) / 2)]) as hdul:
            exts = len(hdul)
            for hdu in hdul:
                testim = hdu.data
                testim[testim == testim[0, 0]] = np.nan
                vmin.append(
                    np.nanmean(testim) - 3 * np.nanstd(testim)
                )
                vmax.append(
                    np.nanmean(testim) + 3 * np.nanstd(testim)
                )

        """BLOCK: Figure setup"""
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(nrows=3, ncols=3)
        ax_im = fig.add_subplot(gs[:, :2])
        ax_see = fig.add_subplot(gs[1, 2])
        ax_lc = fig.add_subplot(gs[2, 2])
        ax_prof = fig.add_subplot(gs[0, 2])

        """BLOCK: Image Panel"""
        im = ax_im.imshow(
            img, interpolation="none", origin="lower", cmap="gray", vmin=vmin[0], vmax=vmax[0]
        )

        """BLOCK: Profile Panel"""
        ax_prof.plot(fts_w, fts_s, c="C0")
        ax_prof.set_xlim(fts_w[0], fts_w[-1])
        ax_prof.set_ylim(0., 1.1)
        ax_prof.set_xlabel("Wavelength [AA]")
        ax_prof.set_title("FTS Atlas Reference Profile", weight="bold")
        prof_vline = ax_prof.axvline(waves[0], ls="-", color="C2", lw=1, zorder=10)

        """BLOCK: Seeing panel"""
        ax_see.plot(ibis_ts, scin, c="C0")
        ax_see.set_ylim(0, 2.5)
        ax_see.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_see.set_ylabel("Scin. [asec]", color="C0", weight="bold")
        ax_see.tick_params(axis="y", labelcolor="C0")
        ax_see.set_xlim(starttime, endtime)
        ax_see.set_title("Scintillation/Light Level", weight="bold")

        ax_see_llvl = ax_see.twinx()
        ax_see_llvl.plot(ibis_ts, llvl, color="C1")
        ax_see_llvl.set_ylabel("Light Level", color="C1", weight="bold", rotation=270, labelpad=25)
        ax_see_llvl.tick_params(axis="y", labelcolor="C1")
        ax_see_llvl.set_ylim(0.5, 5.5)

        see_vline = ax_see.axvline(starttime, ls="-", color="C2", lw=1, zorder=10)

        """BLOCK: Lightcurve panel"""
        ax_lc.plot(lc_ts, lc, c="C0")
        ax_lc.set_xlim(starttime, endtime)
        ax_lc.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_lc.set_ylabel("Line Core LC.", color="C0", weight="bold")
        ax_lc.tick_params(axis="y", labelcolor="C0")
        ax_lc.set_title("Lightcuve(s)", weight="bold")

        lc_vline = ax_lc.axvline(starttime, ls="-", color="C2", lw=1, zorder=10)
        if len(goes_timestamps) > 0:
            ax_goes = ax_lc.twinx()
            ax_goes.plot(goes_timestamps, long, color="C1")
            ax_goes.set_ylim(-9, -2)
            ax_goes.set_yticks([-7.5, -6.5, -5.5, -4.5, -3.5],
                               labels=["A", "B", "C", "M", "X"])
            ax_goes.set_ylabel("GOES LC", weight="bold", color="C1")
            ax_goes.tick_params(axis="y", labelcolor="C1")

        title = fig.suptitle(starttime.astype(str), y=0.95)
        fig.tight_layout()

        def ibis_animate(i):
            try:
                with fits.open(flist[i // exts]) as hdul:
                    hdu = hdul[i % exts]
                    t = np.datetime64(hdu.header["DATE"]) + np.timedelta64(int(1000 * hdu.header["TIME"]), "ms")
                    im.set_array(hdu.data)
                    norm = plt.Normalize(vmin[i%exts], vmax[i%exts])
                    im.set_norm(norm)
                    prof_vline.set_xdata([waves[i% exts], waves[i% exts]])
                    see_vline.set_xdata([t, t])
                    lc_vline.set_xdata([t, t])
                    title.set_text(t.astype(str))
            except (FileNotFoundError, IndexError) as err:
                print(f"Movie could not be made: {err}")
                pass
            return im,

        anim = animation.FuncAnimation(
            fig, ibis_animate,
            frames=tqdm.tqdm(
                range(exts * len(flist)), file=sys.stdout, desc="IBIS "+filt, disable=not progress
            ),
            interval=1000/fps,
            init_func=lambda: None
        )

        anim.save(outfile, fps=fps)

    return


def rosa_hardcam_movie_maker(
        filelist: list, save_dir: str, savename: str, channel: str,
        fps: int=0, vmin: int=-3, vmax: int=3, progress: bool=True
    ) -> None:
    """Creates movie of ROSA or HARDcam Level-1 data

    Parameters
    ----------
    filelist : list
        List of paths to reduced FITS files to make movies from
    save_dir : str
        Path to save output movie
    savename : str
        Name of movie file
    channel : str
        String with the imager channel.
    fps : int, optional
        Optional final FPS. If left at 0, will use best-guess values, default 0
    vmin : int, optional
        vmin for final movie, expressed as mean(middle_frame) + vmin*std(middle_frame), default 3
    vmax : int, optional
        vmax for final movie, expressed as mean(middle_frame) + vmax*std(middle_frame), default 3
    progress : bool, optional
        Whether to spawn tqdm progress bar, default True
    """
    # Clean up channel names from ssosoft.imagers.destretch:
    # ROSA_GBAND -> ROSA GBAND
    channel = channel.replace("_", " ")
    # ZYLA -> HARDcam
    channel = "HARDcam" if channel == "ZYLA" else channel
    with fits.open(filelist[0]) as hdul:
        if "STARTOBS" not in hdul[0].header.keys() and "DATE" not in hdul[0].header.keys():
            startobs = endobs = None
        else:
            if "STARTOBS" in hdul[0].header.keys():
                startobs = np.datetime64(hdul[0].header["STARTOBS"])
            else:
                startobs = np.datetime64(hdul[0].header["DATE"])
            with fits.open(filelist[-1]) as hdul2:
                if "ENDSOBS" in hdul2[0].header.keys():
                    endobs = np.datetime64(hdul2[0].header["ENDOBS"])
                if "STARTOBS" in hdul2[0].header.keys():
                    endobs = np.datetime64(hdul2[0].header["STARTOBS"])
                else:
                    endobs = np.datetime64(hdul2[0].header["DATE"])
        if "SCINT" in hdul[0].header.keys():
            dcss_params = True
        else:
            dcss_params = False
        if "CDELT1" in hdul[0].header.keys():
            extents = [0, hdul[0].data.shape[1] * hdul[0].header["CDELT1"],
                       0, hdul[0].data.shape[0] * hdul[0].header["CDELT2"]]
        else:
            extents = [0, 1, 0, 1]
    if fps == 0:
        if len(filelist) <= 240:
            fps = 6
        else:
            fps = 24
    lc = []
    timestamps = []
    llvl = []
    scin = []
    for file in filelist:
        with fits.open(file) as hdul:
            if file == filelist[0]:
                img = hdul[0].data
            lc.append(np.mean(hdul[0].data[hdul[0].data != hdul[0].data[0, 0]]))
            if startobs is not None:
                timestamps.append(
                    np.datetime64(hdul[0].header["STARTOBS"]) if "STARTOBS" in hdul[0].header.keys() else
                    np.datetime64(hdul[0].header["DATE"])
                )
            else:
                timestamps.append(filelist.index(file))
            if dcss_params:
                llvl.append(hdul[0].header["LLVL"])
                scin.append(hdul[0].header["SCIN"])
    with fits.open(filelist[int(len(filelist) / 2)]) as hdul:
        vmin = np.nanmean(hdul[0].data) - vmin * np.nanstd(hdul[0].data)
        vmax = np.nanmean(hdul[0].data) + vmax * np.nanstd(hdul[0].data)
    if startobs is not None:
        goes_timestamps, short, long = get_goes_timeseries(startobs, endobs, save_dir)
    else:
        goes_timestamps = short = long = []

    """Create Figure Block"""
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(nrows=2, ncols=3)
    ax_im = fig.add_subplot(gs[:, :2])
    ax_lc = fig.add_subplot(gs[1, 2])
    if dcss_params:
        ax_see = fig.add_subplot(gs[1, 2])
    else:
        ax_see = None

    """BLOCK: Image Panel"""
    im = ax_im.imshow(
        img, interpolation="none", origin="lower", cmap="gray", vmin=vmin, vmax=vmax
    )
    if extents != [0, 1, 0, 1]:
        ax_im.set_ylabel("Extent [arcsec]")
        ax_im.set_xlabel("Extent [arcsec]")
    if dcss_params and ax_see is not None:
        """BLOCK: Seeing panel"""
        ax_see.plot(timestamps, scin, c="C0")
        ax_see.set_ylim(0, 2.5)
        ax_see.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_see.set_ylabel("Scin. [asec]", color="C0", weight="bold")
        ax_see.tick_params(axis="y", labelcolor="C0")
        ax_see.set_xlim(timestamps[0], timestamps[-1])
        ax_see.set_title("Scintillation/Light Level", weight="bold")

        ax_see_llvl = ax_see.twinx()
        ax_see_llvl.plot(timestamps, llvl, color="C1")
        ax_see_llvl.set_ylabel("Light Level", color="C1", weight="bold", rotation=270, labelpad=25)
        ax_see_llvl.tick_params(axis="y", labelcolor="C1")
        ax_see_llvl.set_ylim(0.5, 6.0)

        see_vline = ax_see.axvline(timestamps[0], ls="-", color="C2", lw=1, zorder=10)

    """BLOCK: Lightcurve panel"""
    ax_lc.plot(timestamps, lc, c="C0")
    if startobs is not None:
        ax_lc.set_xlim(startobs, endobs)
        ax_lc.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        ax_lc.set_xlim(timestamps[0], timestamps[-1])
    ax_lc.set_ylabel("Zyla LC.", color="C0", weight="bold")
    ax_lc.tick_params(axis="y", labelcolor="C0")
    ax_lc.set_title("Lightcuve(s)", weight="bold")

    lc_vline = ax_lc.axvline(timestamps[0], ls="-", color="C2", lw=1, zorder=10)
    if len(goes_timestamps) > 0:
        ax_goes = ax_lc.twinx()
        ax_goes.plot(goes_timestamps, long, color="C1")
        ax_goes.set_ylim(-9, -2)
        ax_goes.set_yticks([-7.5, -6.5, -5.5, -4.5, -3.5],
                            labels=["A", "B", "C", "M", "X"])
        ax_goes.set_ylabel("GOES LC", weight="bold", color="C1")
        ax_goes.tick_params(axis="y", labelcolor="C1")

    title = fig.suptitle(f"{channel}: {str(timestamps[0])}", y=0.95)
    fig.tight_layout()

    def animate_imager(i):
        try:
            with fits.open(filelist[i]) as hdul:
                if startobs is not None:
                    t = np.datetime64(hdul[0].header["STARTOBS"]) if "STARTOBS" in hdul[0].header.keys() else \
                        np.datetime64(hdul[0].header["DATE"])
                else:
                    t = i
                im.set_array(hdul[0].data)
                if startobs is not None and dcss_params:
                    see_vline.set_xdata([t, t])
                lc_vline.set_xdata([t, t])
                title.set_text(f"{channel}: {str(t)}")
        except (FileNotFoundError, IndexError) as err:
            print(f"Movie could not be made: {err}")
            pass
        return im,

    anim = animation.FuncAnimation(
        fig, animate_imager,
        frames=tqdm.tqdm(
            range(len(filelist)),
            file=sys.stdout,
            disable=not progress,
            desc=f"Creating {channel} movie..."
        ),
        interval=1000 / fps,
        init_func=lambda: None
    )

    anim.save(os.path.join(save_dir, savename), fps=fps)

    return

def get_goes_timeseries(starttime, endtime, outpath):
    """

    Parameters
    ----------
    starttime
    endtime
    outpath

    Returns
    -------

    """
    trange = a.Time(starttime.astype(str), endtime.astype(str))
    search = Fido.search(trange, a.Instrument("XRS"))
    if search.file_num == 0:
        goes_timestamps = []
        short = []
        long = []
    else:
        search = search[0, 0]
        goes_dl = Fido.fetch(search, path=outpath)
        goes_timeseries = ts.TimeSeries(goes_dl)
        goes_data = goes_timeseries.to_dataframe()
        goes_data = goes_data[(goes_data["xrsa_quality"] == 0) & (goes_data["xrsb_quality"] == 0)]
        goes_timeseries = ts.TimeSeries(goes_data, goes_timeseries.meta, goes_timeseries.units)
        truncated = goes_timeseries.truncate(starttime.astype(str), endtime.astype(str))
        if len(goes_data) == 0:
            goes_timestamps = []
            short = []
            long = []
        else:
            goes_timestamps = truncated.time.value.astype("datetime64[s]")
            short = np.log10(truncated.quantity("xrsa").value)
            long = np.log10(truncated.quantity("xrsb").value)
    return goes_timestamps, short, long

