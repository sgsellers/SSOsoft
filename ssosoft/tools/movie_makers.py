import glob
import os
import sys

import astropy.io.fits as fits
import matplotlib.animation as animation
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.colors import Normalize
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
                    norm = Normalize(vmin[i%exts], vmax[i%exts])
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
                if "SCIN" in hdul[0].header.keys():
                    scin.append(hdul[0].header["SCIN"])
                elif "SCINT" in hdul[0].header.keys():
                    scin.append(hdul[0].header["SCINT"])
    with fits.open(filelist[int(len(filelist) / 2)]) as hdul:
        vmin = np.nanmean(hdul[0].data) + vmin * np.nanstd(hdul[0].data)
        vmax = np.nanmean(hdul[0].data) + vmax * np.nanstd(hdul[0].data)
    if startobs is not None:
        goes_timestamps, short, long = get_goes_timeseries(startobs, endobs, save_dir)
    else:
        goes_timestamps = short = long = []

    """Create Figure Block"""
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(nrows=2, ncols=3)
    ax_im = fig.add_subplot(gs[:, :2])
    ax_lc = fig.add_subplot(gs[0, 2])
    if dcss_params:
        ax_see = fig.add_subplot(gs[1, 2])
    else:
        ax_see = None

    """BLOCK: Image Panel"""
    im = ax_im.imshow(
        img, interpolation="none",
        origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
        extent=extents
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
    ax_lc.set_ylabel(f"{channel} LC.", color="C0", weight="bold")
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

def spinor_movie_maker(
        spinor_filename: str, field_images: np.ndarray | None,
        savedir: str, savename: str, central_wavelength: float,
        fps: int=0, progress: bool=True
):
    """Creates movie from SPINOR raster

    Parameters
    ----------
    spinor_filename : str
        Reduced SPINOR file. We'll want the reduced data and SCIN/LLVL params in the file 
    field_images : np.ndarray | None
        IQUV images of user-selected line
    savedir : str
        Location to save context movie
    savename : str
        Filename for context movie
    central_wavelength : float
        Center/Most important wavelength
    fps : int, optional
        FPS of final movie. If 0, use best guess, by default 0
    progress : bool, optional
        If True, spawns a progress bar, by default True
    """
    with fits.open(spinor_filename) as hdul:
        scintillation = hdul["METADATA"].data["TELESCIN"]
        timedeltas = [np.timedelta64(int(i * 1000), "ms") for i in hdul["METADATA"].data["T_ELAPSED"]]
        timestamps = np.datetime64(hdul["METADATA"].header["TRPOS1"]) + timedeltas
        goes_timestamps, short, long = get_goes_timeseries(timestamps[0], timestamps[-1], savedir)
        if field_images is None:
            iimg = np.mean(hdul["STOKES-I"].data, axis=-1)
            ilabel = "Continuum Stokes-I"
            qimg = np.mean(np.abs(hdul["STOKES-I"].data), axis=-1)
            qlabel = "Mean |Stokes-Q|"
            uimg = np.mean(np.abs(hdul["STOKES-U"].data), axis=-1)
            ulabel = "Mean |Stokes-U|"
            vimg = np.mean(np.abs(hdul["STOKES-V"].data), axis=-1)
            vlabel = "Mean |Stokes-V|"

            contimg = None
        else:
            iimg = field_images[0]
            ilabel = "Line-Core Stokes-I"
            qimg = field_images[1]
            qlabel = "Line-Integrated Stokes-Q"
            uimg = field_images[2]
            ulabel = "Line-Integrated Stokes-U"
            vimg = field_images[3]
            vlabel = "Line-Core Deriv. Stokes-V"

            contimg = np.mean(hdul["STOKES-I"].data, axis=-1)

        fovx = hdul["STOKES-I"].header["CDELT1"] * iimg.shape[1]
        fovy = hdul["STOKES-I"].header["CDELT2"] * iimg.shape[0]
        xextent = np.linspace(0, fovx, num=iimg.shape[1]) + hdul["STOKES-I"].header["CDELT1"]/2
        wavegrid = hdul["lambda-coordinate"].data

        # Figure setup
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(nrows=3, ncols=4)
        if contimg is None:
            ax_goes = fig.add_subplot(gs[2, :])
            ax_cont = None
        else:
            ax_goes = fig.add_subplot(gs[2, 1:])
            ax_cont = fig.add_subplot(gs[2, 0])
        axi = fig.add_subplot(gs[0, 0])
        axq = fig.add_subplot(gs[0, 1])
        axu = fig.add_subplot(gs[1, 0])
        axv = fig.add_subplot(gs[1, 1])

        axi_slit = fig.add_subplot(gs[0, 2])
        axq_slit = fig.add_subplot(gs[0, 3])
        axu_slit = fig.add_subplot(gs[1, 2])
        axv_slit = fig.add_subplot(gs[1, 3])

        # Field image plotting
        axi.imshow(iimg, origin="lower", cmap="gray",
                   vmin=np.mean(iimg) - 3*np.std(iimg),
                   vmax=np.mean(iimg) + 3*np.std(iimg),
                   extent=(0, fovx, 0, fovy), aspect="auto")
        axq.imshow(qimg, origin="lower", cmap="gray",
                   vmin=np.mean(qimg) - 1.5*np.std(qimg),
                   vmax=np.mean(qimg) + 1.5*np.std(qimg),
                   extent=(0, fovx, 0, fovy), aspect="auto")
        axu.imshow(uimg, origin="lower", cmap="gray",
                   vmin=np.mean(uimg) - 1.5*np.std(uimg),
                   vmax=np.mean(uimg) + 1.5*np.std(uimg),
                   extent=(0, fovx, 0, fovy), aspect="auto")
        axv.imshow(vimg, origin="lower", cmap="gray",
                   vmin=np.mean(vimg) - 2*np.std(vimg),
                   vmax=np.mean(vimg) + 2*np.std(vimg),
                   extent=(0, fovx, 0, fovy), aspect="auto")

        axi.set_title(ilabel, weight="bold")
        axq.set_title(qlabel, weight="bold")
        axu.set_title(ulabel, weight="bold")
        axv.set_title(vlabel, weight="bold")
        axi.set_ylabel("Extent [asec]")
        axu.set_ylabel("Extent [asec]")
        axu.set_xlabel("Extent [asec]")
        axv.set_xlabel("Extent [asec]")

        if ax_cont is not None:
            ax_cont.imshow(contimg, origin="lower", cmap="gray",
                           vmin=np.mean(contimg) - 3*np.std(contimg),
                           vmax=np.mean(contimg) + 3*np.std(contimg),
                           extent=(0, fovx, 0, fovy), aspect="auto")
            ax_cont.set_title("Continuum Stokes-I", weight="bold")
            ax_cont.set_ylabel("Extent [asec]")
            ax_cont.set_xlabel("Extent [asec]")

        iline = axi.axvline(xextent[0], c="C1", ls="-", lw=1, zorder=10)
        qline = axq.axvline(xextent[0], c="C1", ls="-", lw=1, zorder=10)
        uline = axu.axvline(xextent[0], c="C1", ls="-", lw=1, zorder=10)
        vline = axv.axvline(xextent[0], c="C1", ls="-", lw=1, zorder=10)
        if ax_cont is not None:
            cline = ax_cont.axvline(xextent[0], c="C1", ls="-", lw=1, zorder=10)
        else:
            cline = None

        # Slit image plotting
        islit = hdul["STOKES-I"].data[:, 0, :]
        qslit = hdul["STOKES-Q"].data[:, 0, :]
        uslit = hdul["STOKES-U"].data[:, 0, :]
        vslit = hdul["STOKES-V"].data[:, 0, :]

        i_s = axi_slit.imshow(islit, origin="lower", cmap="gray",
                             vmin=np.mean(islit) - 3*np.std(islit),
                             vmax=np.mean(islit) + 3*np.std(islit),
                             extent=(0, fovy, wavegrid[0], wavegrid[-1]),
                             aspect="auto")
        q_s = axq_slit.imshow(qslit, origin="lower", cmap="gray",
                             vmin=np.mean(qslit) - 2*np.std(qslit),
                             vmax=np.mean(qslit) + 2*np.std(qslit),
                             extent=(0, fovy, wavegrid[0], wavegrid[-1]),
                             aspect="auto")
        u_s = axu_slit.imshow(uslit, origin="lower", cmap="gray",
                             vmin=np.mean(uslit) - 2*np.std(uslit),
                             vmax=np.mean(uslit) + 2*np.std(uslit),
                             extent=(0, fovy, wavegrid[0], wavegrid[-1]),
                             aspect="auto")
        v_s = axv_slit.imshow(vslit, origin="lower", cmap="gray",
                             vmin=np.mean(vslit) - 2*np.std(vslit),
                             vmax=np.mean(vslit) + 2*np.std(vslit),
                             extent=(0, fovy, wavegrid[0], wavegrid[-1]),
                             aspect="auto")

        axi_slit.set_title("Stokes-I", weight="bold")
        axi_slit.set_ylabel("Extent [asec]")
        axq_slit.set_title("Stokes-Q", weight="bold")
        axu_slit.set_title("Stokes-U", weight="bold")
        axu_slit.set_ylabel("Extent [asec]")
        axu_slit.set_xlabel("Wavelength [$\\AA$]")
        axv_slit.set_title("Stokes-V", weight="bold")
        axv_slit.set_xlabel("Wavelength [$\\AA$]")


        # GOES/Scintillation plot
        ax_goes.plot(timestamps, scintillation, c="C0")
        ax_goes.set_xlim(timestamps[0], timestamps[-1])
        ax_goes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_goes.set_ylabel("Scintillation", color="C0", weight="bold")
        ax_goes.tick_params(axis="y", labelcolor="C0")
        ax_goes.set_title("Telescope Scintillation/GOES Lightcuve", weight="bold")

        lc_vline = ax_goes.axvline(timestamps[0], ls="-", color="C2", lw=1, zorder=10)
        if len(goes_timestamps) > 0:
            ax_goes2 = ax_goes.twinx()
            ax_goes2.plot(goes_timestamps, long, color="C1")
            ax_goes2.set_ylim(-9, -2)
            ax_goes2.set_yticks([-7.5, -6.5, -5.5, -4.5, -3.5],
                                labels=["A", "B", "C", "M", "X"])
            ax_goes2.set_ylabel("GOES LC", weight="bold", color="C1")
            ax_goes2.tick_params(axis="y", labelcolor="C1")

        title = fig.suptitle(f"SPINOR {central_wavelength}: {str(timestamps[0])}", weight="bold")
        fig.tight_layout()

        def animate_spectropolarimeter(i):
            # Update line positions
            iline.set_xdata([xextent[i], xextent[i]])
            qline.set_xdata([xextent[i], xextent[i]])
            uline.set_xdata([xextent[i], xextent[i]])
            vline.set_xdata([xextent[i], xextent[i]])
            if ax_cont is not None:
                cline.set_xdata([xextent[i], xextent[i]])
            lc_vline.set_xdata([timestamps[i], timestamps[i]])
            # Update Slit Images
            islit = hdul["STOKES-I"].data[:, i, :]
            qslit = hdul["STOKES-Q"].data[:, i, :]
            uslit = hdul["STOKES-U"].data[:, i, :]
            vslit = hdul["STOKES-V"].data[:, i, :]
            i_s.set_array(islit)
            q_s.set_array(qslit)
            u_s.set_array(uslit)
            v_s.set_array(vslit)
            inorm = Normalize(np.mean(islit) - 3*np.std(islit))
            i_s.set_norm(inorm)
            qnorm = Normalize(np.mean(qslit) - 2*np.std(qslit))
            q_s.set_norm(qnorm)
            unorm = Normalize(np.mean(uslit) - 2*np.std(uslit))
            u_s.set_norm(unorm)
            vnorm = Normalize(np.mean(vslit) - 2*np.std(vslit))
            v_s.set_norm(vnorm)
            title.set_text(f"SPINOR {central_wavelength}: {str(timestamps[i])}")
            return

        if fps == 0:
            # Best guess: We want to start around 6 fps
            # 6 fps gives a maximum 1.5 minute movie for a 550-step map
            # Want a minimum movie length of 30 seconds
            # So slower if the desired movie is < 30 seconds
            if iimg.shape[1] / 8 < 30:
                fps = iimg.shape[1] / 30
            else:
                fps = 6
        anim = animation.FuncAnimation(
            fig, animate_spectropolarimeter,
            frames=tqdm.tqdm(
                range(iimg.shape[1]),
                file=sys.stdout,
                disable=not progress,
                desc="Creating SPINOR movie..."
            ),
            interval=1000 / fps,
            init_func=lambda: None
        )

        anim.save(os.path.join(savedir, savename), fps=fps)
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

