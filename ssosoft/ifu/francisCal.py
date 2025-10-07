import configparser
import glob
import logging
import os
from importlib import resources

import astropy.io.fits as fits
import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as scinteg
import scipy.interpolate as scinterp
import scipy.ndimage as scind
import scipy.optimize as scopt
import tqdm
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames

from ssosoft.ifu import IFUTools as ifu
from ssosoft.spectral import polarimetryTools as pol
from ssosoft.spectral import spectraTools as spex


class FrancisCal:
    """
    The Sunspot Solar Observatory Consotrium's software for reducing FRANCIS
    (The Fibre-Resolved opticAl and Near-ultraviolet Czerny-turner Imaging Spectropolarimeter)
    data from the Dunn Solar Telescope. Currently, FRANCIS is operating in imaging
    mode ONLY (no polarimetry). If I have time, I will add a skeleton for the polarimetry
    as I understand it will be structured.

    """
    from ssosoft import ssosoftConfig

    IMAGESHAPE = (2048, 2048)
    PIXEL_SIZE = 11 #um
    APPROX_SPECTRAL_PIXEL = {
            2400 : 0.0072 * PIXEL_SIZE, # Angstrom/pixel
            3600 : 0.0045 * PIXEL_SIZE,
            4320 : 0.0032 * PIXEL_SIZE
        }
    FIBER_APERTURE = 40 # um
    FIBER_WIDTH = 55 # um, with cladding
    PLATE_SCALE = 22.5789 # arcsec/mm
    VSPACING = FIBER_WIDTH * np.sin(45 * np.pi/180.) # um
    HSPACING_ASEC = 0.001 * FIBER_WIDTH * PLATE_SCALE
    VSPACING_ASEC = 0.001 * VSPACING * PLATE_SCALE
    FIBER_APERTURE_ASEC = 0.001 * FIBER_APERTURE * PLATE_SCALE

    def __init__(self, wavelength: str, config_file: str) -> None:
        """Class initialization

        Parameters
        ----------
        wavelength : str
            Wavelength key to use for config file. E.g., "CaK", "BaII", "FeI"...
        config_file : str
            Path to configuration file
        """
        try:
            f = open(config_file, "r")
            f.close()
        except Exception as err:
            print("Exception: {0}".format(err))
            raise

        self.config_file = config_file
        self.wavelength = wavelength
        self.mode = "SPECTRAL" # or "POLARIMETRY"

        self.date = ""
        self.time = ""

        self.science_file_list = []
        self.science_base = ""
        self.flat_base = ""
        self.dark_base = ""
        self.lamp_flat_base = ""
        self.lamp_dark_base = ""
        self.work_base = ""
        self.calibrated_base = ""

        self.data_file_pattern = ""
        self.dark_file_pattern = ""
        self.flat_file_pattern = ""
        self.lamp_flat_file_pattern = ""
        self.lamp_dark_file_pattern = ""

        self.reduced_file_pattern = ""

        self.logfile = ""

        self.logger = logging.getLogger()

        self.progress = False
        self.plot = False

        self.fiber_map_path = ""
        self.fiber_map_filename = ""
        self.fiber_map_1d = np.empty(0)
        self.fiber_map_2d = np.empty(0)
        self.fiber_map_skews = np.empty(0)
        self.fiber_map_vskews = np.empty(0)

        self.fiber_map_1d_tweaked = np.empty(0)
        self.fiber_map_2d_tweaked = np.empty(0)
        self.rigid_xskews = np.empty(0)
        self.rigid_yskews = np.empty(0)

        self.lamp_gain_reduced = ""
        self.solar_gain_reduced = ""

        self.grating_lpmm = 0.
        self.grating_central_wavelength = 0.
        self.camera_fl = 0.

        self.dark = np.empty(0)
        self.solar_flat = np.empty(0)
        self.deskewed_flat = np.empty(0)

        self.lamp_gain = np.empty(0)
        self.coarse_gain_table = np.empty(0)
        self.gain_table = np.empty(0)
        self.flat_deskew = np.empty(0)
        self.francis_line_cores = []
        self.fts_line_cores = []

        self.spectral_pixel = 0

        self.fts_wavelengths = np.empty(0)
        self.fts_spectrum = np.empty(0)

        self.dcss_params = {}
        self.dcss_log = ""

        self.analysis_ranges = None
        self.spectral_analysis = False # Not sure how useful it'll be

        return

    def francis_run_calibration(self) -> None:
        """Function block for FRANCIS calibration"""
        self.francis_configure_run()
        if self.progress:
            print(f"Found {len(self.science_file_list)} science map files in base directory:\n"
                  f"{self.science_base}\n"
                  "Reduced files will be saved to: \n"
                  f"{self.calibrated_base}")
        self.logger.info(
            f"Found {len(self.science_file_list)} science map files in base directory: {self.science_base}"
        )
        self.logger.info(
            "Reduced files will be saved to: {self.calibrated_base}"
        )
        self.parse_dcss()
        self.francis_get_cal_images()
        if self.plot:
            plt.pause(2)
            plt.close("all")
        if self.mode == "POLARIMETY":
            self.reduce_francis_polarimetry_maps()
        else:
            self.reduce_francis_spectral_maps()

        return

    def francis_configure_run(self) -> None:
        """Sets up class variables"""

        def assert_base_dirs(base_dir):
            assert (os.path.isdir(base_dir)), (
                "Directory does not exist: {0}".format(base_dir)
            )

        config = configparser.ConfigParser()
        config.read(self.config_file)

        if "SHARED" in list(config.keys()):
            self.dcss_log = config["SHARED"]["DCSSLog"] if "dcsslog" \
                in config["SHARED"].keys() else self.dcss_log
            self.PLATE_SCALE = config["SHARED"]["plateScale"] if "platescale" \
                in config["SHARED"].keys() else self.PLATE_SCALE

        self.science_base = config[self.wavelength]["dataBase"]
        # If these are undefined, assign them to the same directory as dataBase
        # Useful for ROSA where all files are in the same directory.
        self.dark_base = config[self.wavelength]["darkBase"] if "darkbase" \
            in config[self.wavelength].keys() else self.science_base
        self.flat_base = config[self.wavelength]["flatBase"] if "flatbase" \
            in config[self.wavelength].keys() else self.science_base
        self.lamp_flat_base = config[self.wavelength]["lampflatBase"] if "lampflatbase" \
            in config[self.wavelength].keys() else ""
        self.lamp_dark_base = config[self.wavelength]["lampdarkBase"] if "lampdarkbase" \
            in config[self.wavelength].keys() else ""

        # If everything is in its own directory, having each in the config file is silly
        self.data_file_pattern = config[self.wavelength]["dataFilePattern"]
        self.dark_file_pattern = config[self.wavelength]["darkFilePattern"] if \
            "darkfilepattern" in config[self.wavelength].keys() else self.data_file_pattern
        self.flat_file_pattern = config[self.wavelength]["flatFilePattern"] if \
            "flatfilepattern" in config[self.wavelength].keys() else self.data_file_pattern
        self.lamp_flat_file_pattern = config[self.wavelength]["lampflatFilePattern"] if \
            "lampflatfilepattern" in config[self.wavelength].keys() else self.data_file_pattern
        self.lamp_dark_file_pattern = config[self.wavelength]["lampdarkFilePattern"] if \
            "lampdakrfilepattern" in config[self.wavelength].keys() else self.data_file_pattern

        self.work_base = config[self.wavelength]["workBase"]

        self.date = config[self.wavelength]["obsDate"]
        self.time = config[self.wavelength]["obsTime"]

        self.reduced_file_pattern = config[self.wavelength]["reducedFilePattern"]

        self.calibrated_base = os.path.join(self.work_base, "calibrated")
        if not os.path.isdir(self.calibrated_base):
            print("{0}: os.mkdir: attempting to create directory:"
                    "{1}".format(__name__, self.calibrated_base)
                    )
            try:
                os.mkdir(self.calibrated_base)
            except Exception as err:
                print("An exception was raised: {0}".format(err))
                raise

        self.plot = config[self.wavelength]["plot"] if "plot" in config[self.wavelength].keys() else "False"
        if "t" in self.plot.lower():
            self.plot = True
        else:
            self.plot = False
        self.progress = config[self.wavelength]["progress"] if "progress" \
            in config[self.wavelength].keys() else "False"
        if "t" in self.progress.lower():
            self.progress = True
        else:
            self.progress = False

        self.logfile = os.path.join(self.work_base, f"{self.time}_FRANCIS_{self.wavelength}.log")
        logging.config.fileConfig(self.config_file, defaults={"logfilename": self.logfile})
        self.logger = logging.getLogger(f"FRANCIS_{self.wavelength}_Log")
        # Intro message
        self.logger.info(f"This is SSOsoft version {self.ssosoftConfig.__version__}")
        self.logger.info(f"Contact {self.ssosoftConfig.__email__} to report bugs, make suggestions, "
                         "of contribute")
        self.logger.info(f"Now configuring this FRANCIS {self.wavelength} data calibration run.")

        try:
            assert_base_dirs(self.dark_base)
        except AssertionError as err:
            self.logger.critical("Fatal: {0}".format(err))
            raise
        else:
            self.logger.info("Using dark directory: {0}".format(self.dark_base))

        try:
            assert_base_dirs(self.science_base)
        except AssertionError as err:
            self.logger.critical("Fatal: {0}".format(err))
            raise
        else:
            self.logger.info("Using data directory: {0}".format(self.science_base))

        try:
            assert_base_dirs(self.flat_base)
        except AssertionError as err:
            self.logger.critical("Fatal: {0}".format(err))
            raise
        else:
            self.logger.info("Using flat directory: {0}".format(self.flat_base))

        self.science_file_list = sorted(glob.glob(os.path.join(self.science_base, self.data_file_pattern)))
        self.solar_gain_reduced = os.path.join(self.work_base, "FRANCIS_SOLARGAIN.fits")

        return

    def average_image_from_list(self, filelist: list) -> np.ndarray:
        """Computers average image from list of files"""

        def print_average_image_progress():
            if not fnum % 25:
                self.logger.info("Progress: {:0.1%}".format(fnum/num_img))

        self.logger.info(f"Computing average image from {len(filelist)} files "
                         f"in directory: {os.path.dirname(filelist[0])}")
        avg_img = np.zeros(self.IMAGESHAPE)
        fnum = 0
        num_img = len(filelist)
        for file in tqdm.tqdm(
            filelist,
            desc=f"Computing average image from {len(filelist)} files",
            disable=not self.progress
        ):
            with fits.open(file) as hdul:
                avg_img += np.sum(hdul[0].data, axis=0) # Currently has shape (1, 2048, 2048), may change with pol
                fnum += hdul[0].data.shape[0]
            print_average_image_progress()
        avg_img /= fnum

        self.logger.info(f"Average complete, directory: {os.path.dirname(filelist[0])}")

        return avg_img

    def read_fiber_map(self, path: str="", fname: str="") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reads fiber map and returns positions and weights for fibers
        If path and fname are not given, uses default maps included in
        SSOsoft, computed from calibration data taken 2025-06-29 during
        instrument comissioning.

        Parameters
        ----------
        path : str, optional
            Path to alternate fiber bundle map files, by default ""
        fname : str, optional
            Name of alternate fiber bundle map files, by default ""

        Returns
        -------
        fibermap_2d : np.ndarray
            Map of fibers packed into 20x20 grid. Shape (2, 7, 20, 20)
            The profile of a single fiber can be extracted as:
            yrange_on_chip = fibermap_2d[0, :, y, x]
            fiber_profile = fibermap_2d[1, :, y, x]
        fibermap_1d : np.ndarray
            Map of fibers ungridded. Shape (2, 7, 400)
        skews : np.ndarray
            Skew along slit. Shape (2048,)
        """
        def read_data(datapath, filename) -> fits.hdu.hdulist.HDUList:
            with resources.path(datapath, filename) as df:
                return fits.open(df)

        if path == fname == "":
            mapfile = read_data("ssosoft.ifu.fiber_maps", "fiber_map_20250629_404.656.fits")
            self.logger.info(
                "Attempting to load default fiber map and initial skews."
            )
        else:
            mapfile = fits.open(os.path.join(path, fname))
            self.logger.info(
                f"Loading fiber map {fname} from {path}"
            )
        try:
            fibermap_2d = mapfile[1].data
            fibermap_1d = mapfile[2].data
            skews = mapfile[3].data
            vskews = mapfile[4].data
            mapfile.close()
        except (FileNotFoundError, IndexError) as err:
            raise("Improperly formatted fiber map file", err)
        else:
            self.fiber_map_2d = fibermap_2d
            self.fiber_map_1d = fibermap_1d
            self.fiber_map_skews = skews
            self.fiber_map_vskews = vskews
            self.fiber_map_filename = "fiber_map_20250629_404.656.fits"
            return

    def francis_get_cal_images(self) -> None:
        """Loads or creates flat, dark, solar gain, and polcal arrays"""
        self.francis_get_flat_dark()
        self.francis_get_lamp_flat_dark()
        self.read_fiber_map()
        self.francis_get_solar_gain()
        if self.mode == "POLARIMETRY":
            self.francis_get_polcal()

        return

    def francis_get_flat_dark(self) -> None:
        """Loads or creates solar flat and dark"""
        if os.path.exists(self.solar_gain_reduced):
            self.logger.info(f"Average flat/dark file found: {self.solar_gain_reduced}")
            self.logger.info("Reading average flat and dark.")
            with fits.open(self.solar_gain_reduced) as hdul:
                self.solar_flat = hdul["SOLAR-FLAT"].data
                self.dark = hdul["SOLAR-DARK"].data
                self.grating_lpmm = hdul[0].header["GRATING"]
                self.grating_central_wavelength = hdul[0].header["GCWVL"]
                self.camera_fl = hdul[0].header["CAMERA"]
        else:
            self.logger.info("Creating average flat and dark from raw files")
            flat_file_list = sorted(glob.glob(os.path.join(self.flat_base, self.flat_file_pattern)))
            dark_file_list = sorted(glob.glob(os.path.join(self.dark_base, self.dark_file_pattern)))
            self.solar_flat = self.average_image_from_list(flat_file_list)
            self.dark = self.average_image_from_list(dark_file_list)
            with fits.open(flat_file_list[0]) as hdul:
                grating_lpmm = hdul[0].header["PI SPECTROMETER GRATING SELECTED"]
                # lpmm info is in format [name, lpmm][0][1]
                self.grating_lpmm = float(grating_lpmm.split("]")[0].split(",")[1])
                self.camera_fl = float(hdul[0].header["PI SPECTROMETER CALIBRATIONINFORMATION FOCALLENGTH"])
                self.grating_central_wavelength = 10 * float(
                    hdul[0].header["PI SPECTROMETER GRATING CENTERWAVELENGTH"]
                )
        return

    def francis_get_lamp_flat_dark(self) -> None:
        """Loads or creates lamp flat and dark
        As FRANCIS does not currently do lamp flats, this will default
        to setting the self.lamp_gain array to all ones."""
        if os.path.exists(self.lamp_gain_reduced):
            self.logger.info(f"Average lamp gain file found: {self.lamp_gain_reduced}")
            self.logger.info("Reading lamp gain.")
            with fits.open(self.lamp_gain_reduced) as hdul:
                self.lamp_gain = hdul[0].data
        elif os.path.exists(self.lamp_flat_base) and os.path.exists(self.lamp_dark_base):
            self.logger.info("Creating lamp gain from raw files.")
            lflat_files = glob.glob(os.path.join(self.lamp_flat_base, self.lamp_flat_file_pattern))
            ldark_files = glob.glob(os.path.join(self.lamp_dark_base, self.lamp_dark_file_pattern))
            lflat = self.average_image_from_list(lflat_files)
            ldark = self.average_image_from_list(ldark_files)
            self.lamp_gain = (lflat - ldark) / np.median(lflat - ldark)
            hdu = fits.PrimaryHDU(self.lamp_gain)
            hdu.header["DATE"] = np.datetime64("now").astype(str)
            fits.HDUList([hdu]).writeto(self.lamp_gain_reduced, overwrite=True)
        else:
            self.logger.warning("No lamp flat or dark files available for observing series.")
            self.logger.warning("Proceeding without.")
            self.lamp_gain = np.ones(self.dark.shape)
        return

    def francis_get_solar_gain(self) -> None:
        """Loads or creates a solar gain table"""
        if os.path.exists(self.solar_gain_reduced):
            self.logger.info(f"Solar gain file found: {self.solar_gain_reduced}")
            with fits.open(self.solar_gain_reduced) as hdul:
                self.deskewed_flat = hdul["DESKEW-FLAT"].data
                self.wavelength_array = hdul["WAVE-ARRAY"].data
                self.coarse_gain_table = hdul["COARSE-GAIN"].data
                self.gain_table = hdul["GAIN"].data
                self.spectral_gain = hdul["SPEC-GAIN"].data
                self.rigid_xskews = hdul["XSKEWS"].data
                self.rigid_yskews = hdul["YSKEWS"].data
                self.fiber_map_1d_tweaked = hdul["FIBER-1D"].data
                self.fiber_map_2d_tweaked = hdul["FIBER-2D"].data
                self.francis_line_cores = [hdul[0].header["LC1"], hdul[0].header["LC2"]]
                self.fts_line_cores = [hdul[0].header["FTSLC1"], hdul[0].header["FTSLC2"]]
        else:
            self.logger.info("Attempting to create gain tables.")
            self.francis_create_solar_gain()
            self.francis_save_gaintables()
        if self.plot:
            self.francis_plot_gaintables()

        return

    def francis_create_solar_gain(self) -> None:
        """
        Creates gain table from mean flat.

        This is done in three phases:
        1.) Determine skewed-ness of flat
        2.) Determine center-averaged profile and do shift/divide
            to create coarse gain table
        3.) Determine nearest-neighbor averaged profile and do shift/divide
            to create first fine gain table
        4.) Fine-tune gain table to include D. Jess method of continuum fitting
            with an arbitrary polynomial.
        """
        init_deskew_flat = np.zeros(self.solar_flat.shape)
        for i in range(init_deskew_flat.shape[0]):
            init_deskew_flat[i, :] = scind.shift(
                (self.solar_flat - self.dark)[i],
                self.fiber_map_skews[i],
                mode="nearest"
            )
        for j in range(init_deskew_flat.shape[1]):
            init_deskew_flat[:, j] = scind.shift(
                init_deskew_flat[:, j], self.fiber_map_vskews[j],
                mode="nearest"
            )
        # Use 5 middle fibers to come up with a mean profile.
        # We'll use this to both create our coarse gain table
        # and get a wavelength grid. We'll want to avoid dead
        # fibers in the bundle, so skip any rows that are all-zero
        center_mean_profile = np.zeros(init_deskew_flat.shape[1])
        good_fiber_counter = 0
        row_counter = 0
        skew_value = 0
        self.logger.info("Performing initial deskew on average flat")
        while good_fiber_counter < 5:
            fiber_xrange = self.fiber_map_1d[0, :, self.fiber_map_1d.shape[2]//2 - 3 + row_counter].astype(int)
            if not all(fiber_xrange == 0): # No profile
                center_mean_profile += np.sum(init_deskew_flat[fiber_xrange], axis=0)
                good_fiber_counter += 1
                skew_value += np.mean(self.fiber_map_skews[fiber_xrange])
            row_counter += 1
        skew_value /= 5
        center_mean_profile /= np.median(center_mean_profile)
        approx_spectral_pixel = self.APPROX_SPECTRAL_PIXEL[int(self.grating_lpmm)]
        approx_wavegrid = np.arange(
            -init_deskew_flat.shape[1]//2 * approx_spectral_pixel,
            init_deskew_flat.shape[1]//2 * approx_spectral_pixel,
            approx_spectral_pixel
        ) + self.grating_central_wavelength
        self.francis_line_cores, self.fts_line_cores, \
            self.fts_wavelengths, self.fts_spectrum = self.francis_fts_line_select(
            approx_wavegrid, center_mean_profile
        )
        fts_line_core_wavelengths = np.array(
            [np.interp(i, np.arange(self.fts_wavelengths.shape[0]), self.fts_wavelengths) for i in self.fts_line_cores]
        )
        self.spectral_pixel = np.abs(np.diff(fts_line_core_wavelengths)) / np.abs(np.diff(self.francis_line_cores))
        zerowvl = fts_line_core_wavelengths[0] - (self.spectral_pixel * self.francis_line_cores[0])
        self.wavelength_array = (np.arange(0, init_deskew_flat.shape[1]) * self.spectral_pixel) + zerowvl
        self.fts_wavelengths, self.fts_spectrum = spex.fts_window(self.wavelength_array[0], self.wavelength_array[-1])
        self.logger.info("Performing fine deskew on average flat, updating fiber map")
        self.deskew_flat(init_deskew_flat, self.francis_line_cores)
        self.logger.info("Creating initial (spectral profile shift-and-divide) gain tables")
        self.gain_table, self.coarse_gain_table = ifu.create_gaintables(
            self.deskewed_flat, self.francis_line_cores[0],
            self.fiber_map_1d_tweaked
        )
        self.logger.info("Creating secondary (FTS atlas residual fitting) gain table")
        self.spectral_gain = ifu.spectral_gain(
            self.deskewed_flat / self.gain_table,
            self.wavelength_array, self.fiber_map_1d_tweaked,
            self.fts_wavelengths, self.fts_spectrum
        )

        return

    def francis_plot_gaintables(self) -> None:
        """If overview plotting is on, pops up an overview of gains"""
        gain_fig = plt.figure("FRANCIS Gain Tables", figsize=(9, 3))
        ax_flat = gain_fig.add_subplot(141)
        ax_deskewed_flat = gain_fig.add_subplot(142)
        ax_combined_gain = gain_fig.add_subplot(143)
        ax_corrected_flat = gain_fig.add_subplot(144)

        ax_flat.imshow(
            self.solar_flat - self.dark,
            origin="lower", cmap="gray",
            vmin=np.mean(self.solar_flat - self.dark) - 2 * np.std(self.solar_flat - self.dark),
            vmax=np.mean(self.solar_flat - self.dark) + 2 * np.std(self.solar_flat - self.dark)
        )
        ax_deskewed_flat.imshow(
            self.deskewed_flat,
            origin="lower", cmap="gray",
            vmin=np.mean(self.deskewed_flat) - 2 * np.std(self.deskewed_flat),
            vmax=np.mean(self.deskewed_flat) + 2 * np.std(self.deskewed_flat)
        )
        ax_combined_gain.imshow(
            self.lamp_gain * self.gain_table * self.spectral_gain,
            origin="lower", cmap="gray",
            vmin=0.5, vmax=1.5
        )
        cflat = self.deskewed_flat / self.lamp_gain / self.gain_table / self.spectral_gain
        ax_corrected_flat.imshow(
            cflat, origin="lower", cmap="gray",
            vmin=np.mean(cflat) - 2 * np.std(cflat),
            vmax=np.mean(cflat) + 2 * np.std(cflat)
        )

        ax_flat.set_title("AVERAGE FLAT")
        ax_deskewed_flat.set_title("DESKEWED FLAT")
        ax_combined_gain.set_title("COMBINED GAIN TABLE")
        ax_corrected_flat.set_title("GAIN-CORRECTED FLAT")
        gain_fig.tight_layout()
        if self.plot:
            filename = os.path.join(self.work_base, "gain_tables.png")
            gain_fig.savefig(filename, bbox_inches="tight")
        plt.show(block=False)
        plt.pause(0.1)

        return

    def francis_save_gaintables(self) -> None:
        """
        Saves gain tables to FITS file.
        Gain file actually includes dark, average flat, deskewed flat,
        coarse gain, fine gain, spectral gain, xskews, yskews, fiber maps, and
        line core locations.
        """
        if os.path.exists(self.solar_gain_reduced):
            self.logger.info(f"File exists: {self.solar_gain_reduced}. Skipping file write.")
            return

        phdu = fits.PrimaryHDU()
        phdu.header["DATE"] = np.datetime64("now").astype(str)
        phdu.header["LC1"] = self.francis_line_cores[0]
        phdu.header["LC2"] = self.francis_line_cores[1]
        phdu.header["FTSLC1"] = self.fts_line_cores[0]
        phdu.header["FTSLC2"] = self.fts_line_cores[1]
        phdu.header["GRATING"] = self.grating_lpmm
        phdu.header["GCWVL"] = self.grating_central_wavelength
        phdu.header["CAMERA"] = self.camera_fl

        flat = fits.ImageHDU(self.solar_flat)
        flat.header["EXTNAME"] = "SOLAR-FLAT"
        dark = fits.ImageHDU(self.dark)
        dark.header["EXTNAME"] = "SOLAR-DARK"
        cgain = fits.ImageHDU(self.coarse_gain_table)
        cgain.header["EXTNAME"] = "COARSE-GAIN"
        fgain = fits.ImageHDU(self.gain_table)
        fgain.header["EXTNAME"] = "GAIN"
        sgain = fits.ImageHDU(self.spectral_gain)
        sgain.header["EXTNAME"] = "SPEC-GAIN"
        xskew = fits.ImageHDU(self.rigid_xskews)
        xskew.header["EXTNAME"] = "XSKEWS"
        yskew = fits.ImageHDU(self.rigid_yskews)
        yskew.header["EXTNAME"] = "YSKEWS"
        fm1 = fits.ImageHDU(self.fiber_map_1d_tweaked)
        fm1.header["EXTNAME"] = "FIBER-1D"
        fm2 = fits.ImageHDU(self.fiber_map_2d_tweaked)
        fm2.header["EXTNAME"] = "FIBER-2D"
        dsk = fits.ImageHDU(self.deskewed_flat)
        dsk.header["EXTNAME"] = "DESKEW-FLAT"
        wva = fits.ImageHDU(self.wavelength_array)
        wva.header["EXTNAME"] = "WAVE-ARRAY"

        hdul = fits.HDUList([phdu, flat, dark, cgain, fgain, sgain, xskew, yskew, fm1, fm2, dsk, wva])
        hdul.writeto(self.solar_gain_reduced, overwrite=True)

        return

    def deskew_flat(self, flat: np.ndarray, line_cores: list) -> None:
        """Performs rigid deskew on flat field images along both axes.
        The skewed-ness contained in the map files is pretty decent, but
        not perfect for each wavelength band. This function is intended to fine-tune
        both axes to get a flat image that is rigidly-straightened along both axes,
        along with the transformations required to place science images on the same
        grid. This will also tweak the fiber maps by performing a center-of-mass
        calculation on each fiber slice in the fiber mass, and adjusting the indices.
        For the deskew along the y-axis, the first selected gain line will be used.
        For the deskew along the x-axis, fiber #220 (index 219) will be the default.
        Fiber index 219 is an isolated fiber (218 and 220 are dead), which provides
        a clear profile. If index 219 ever goes dark, there's a fallback function to
        select a relatively-isolated living fiber.

        Parameters
        ----------
        flat : np.ndarray
            Roughly-deskewed dark-corrected flat
        line_cores : list
            List of user-selected line cores. The first will be used for deskewing

        """
        deskewed_flat = np.zeros(flat.shape)
        default_fiber = 219
        if all(self.fiber_map_1d[0, :, default_fiber] == 0): # Fiber tagged as dead
            fiber_indices = self.select_deskew_fiber()
        else:
            fiber_indices = [
                int(self.fiber_map_1d[0, self.fiber_map_1d.shape[1] // 2, default_fiber] - 5),
                int(self.fiber_map_1d[0, self.fiber_map_1d.shape[1] // 2, default_fiber] + 5)
            ]
        rigid_xshifts = np.zeros(flat.shape[0])
        rigid_yshifts = np.zeros(flat.shape[1])
        vdeskew_slice = flat.copy()[
            int(self.fiber_map_1d[0, 0, 0]):int(self.fiber_map_1d[0, -1, -1]),
            int(line_cores[0]) - 5: int(line_cores[0]) + 6
        ]
        # 2 passes: order 1, then order 2 polynormial
        for i in range(2):
            skews = spex.spectral_skew(vdeskew_slice, order=i+1)
            rigid_xshifts[int(self.fiber_map_1d[0, 0, 0]):int(self.fiber_map_1d[0, -1, -1])] += skews
            for j in range(vdeskew_slice.shape[0]):
                vdeskew_slice[j] = scind.shift(vdeskew_slice[j], skews[j], mode="nearest")
        hdeskew_slice = np.rot90(flat.copy()[fiber_indices[0]:fiber_indices[1]])
        for i in range(2):
            skews = spex.spectral_skew(hdeskew_slice, order=i+1)
            rigid_yshifts += skews[::-1] # Reverse due to rot90 call
            for j in range(hdeskew_slice.shape[0]):
                hdeskew_slice[j] = scind.shift(hdeskew_slice[j], skews[j], mode="nearest")
        for i in range(flat.shape[0]):
            deskewed_flat[i] = scind.shift(flat[i], rigid_xshifts[i], mode="nearest")
        for i in range(flat.shape[1]):
            deskewed_flat[:, i] = scind.shift(flat[:, i], rigid_yshifts[i], mode="nearest")
        self.tweak_fiber_maps(deskewed_flat)
        self.rigid_xskews = rigid_xshifts
        self.rigid_yskews = rigid_yshifts
        self.deskewed_flat = deskewed_flat
        return

    def tweak_fiber_maps(self, deskewed_flat: np.ndarray, niter: int=5) -> None:
        """Generates updated fiber maps from deskewed flat.
        Uses simple center of mass to update fiber centers.


        Parameters
        ----------
        deskewed_flat : np.ndarray
            Flat that has been rigidly deskewed along both axes
        niter : int
            Number of times to adjust centering with CoM calculation
        """
        self.fiber_map_1d_tweaked = np.zeros((5, self.fiber_map_1d.shape[2]))

        # Do the first iteration of CoM based on original map
        for fiber in range(self.fiber_map_1d.shape[2]):
            if not all(self.fiber_map_1d[0, :, fiber] == 0):
                fiber_slice = np.mean(deskewed_flat[self.fiber_map_1d[0, :, fiber].astype(int)], axis=1)
                com = scind.center_of_mass(fiber_slice)[0]
                self.fiber_map_1d_tweaked[:, fiber] = np.arange(com - 2, com + 3) + self.fiber_map_1d[0, 0, fiber]
        # If niter > 1, do another niter - 1 passes
        if niter > 1:
            for iter in range(niter - 1):
                for fiber in range(self.fiber_map_1d.shape[2]):
                    if not all(self.fiber_map_1d_tweaked[:, fiber] == 0):
                        fiber_slice = np.mean(deskewed_flat[self.fiber_map_1d_tweaked[:, fiber].astype(int)], axis=1)
                        com = scind.center_of_mass(fiber_slice)[0]
                        self.fiber_map_1d_tweaked[:, fiber] = np.arange(
                            com - 2, com + 3
                        ) + self.fiber_map_1d_tweaked[0, fiber]
        self.fiber_map_2d_tweaked = self.fiber_map_1d_tweaked.copy().reshape(5, 20, 20)[:, :, ::-1] # Reversed X
        return

    def select_deskew_fiber(self) -> list:
        """Selects an isolated living fiber. Returns indices surrounding it"""

        # Check for living surrounded by dead first. By definition, cannot be first or last fiber
        fiber_index = None
        for i in range(1, self.fiber_map_1d.shape[2] - 1):
            if (
                all(self.fiber_map_1d[0, :, i - 1] == 0) and # Preceding fiber dead
                all(self.fiber_map_1d[0, :, i + 1] == 0) and not # Succeeding fiber dead
                all(self.fiber_map_1d[0, :, i] == 0) # Fiber not dead
            ):
                fiber_index = i
                break
        if fiber_index is not None:
            fiber_indices = [
                int(self.fiber_map_1d[0, self.fiber_map_1d.shape[1]//2, fiber_index] - 5),
                int(self.fiber_map_1d[0, self.fiber_map_1d.shape[1]//2, fiber_index] + 5)
            ]
            return fiber_indices
        # Fallback. Select for isolation
        indices = self.fiber_map_1d[0, 0, :]
        mask = indices == 0
        indices[mask] = np.nan
        fiber_index = np.nanargmax(indices[1:] - indices[:-1])
        fiber_indices = [
            int(self.fiber_map_1d[0, self.fiber_map_1d.shape[1] // 2, fiber_index] - 5),
            int(self.fiber_map_1d[0, self.fiber_map_1d.shape[1] // 2, fiber_index] + 5)
        ]
        return fiber_indices

    @staticmethod
    def francis_fts_line_select(
        wavelength_grid: np.ndarray, average_spectrum: np.ndarray
    ) -> tuple[list, list, np.ndarray, np.ndarray]:
        """User selects corresponding spectral features in FRANCIS and FTS atlas

        Parameters
        ----------
        wavelength_grid : np.ndarray
            FRANCIS wavelength grid
        average_spectrum : np.ndarray
            FRANCIS average spectrum

        Returns
        -------
        francis_line_cores : list
            List of line core positions for FRANCIS
        fts_line_cores : list
            List of line core positions in FTS atlas
        fts_waves : np.ndarray
            Wavelength grid for FTS atlas window
        fts_spec : np.ndarray
            Spectral information for FTS atlas window
        """
        fts_wave, fts_spec = spex.fts_window(wavelength_grid[0], wavelength_grid[-1])
        print("TOP: FRANCIS Spectrum (uncorrected), BOTTOM: FTS Reference Spectrum")
        print("Click to select the same two spectral lines in each plot")
        francis_lines, fts_lines = spex.select_lines_doublepanel(
            average_spectrum,
            fts_spec,
            4
        )
        francis_line_cores = [
            spex.find_line_core(average_spectrum[x - 5:x + 5]) + x - 5 for x in francis_lines
        ]
        fts_line_cores = [
            spex.find_line_core(fts_spec[x - 10:x + 10]) + x - 10 for x in fts_lines
        ]
        return francis_line_cores, fts_line_cores, fts_wave, fts_spec

    def reduce_francis_spectral_maps(self) -> None:
        """
        Main reduction loop for FRANCIS spectral imaging.
        Steps are:
        1.) Read data
        2.) Perform dark subtraction
        3.) Rigid deskew
        4.) Gain correction
        5.) Fiber extraction and placement in grid.
        """
        def print_average_image_progress():
            if not nfile % 25:
                self.logger.info("Main Cal Loop Progress: {:0.1%}".format(nfile/len(self.science_file_list)))

        # FRANCIS' DATE-OBS header keyword is the endobs for the *last* frame.
        # It does not record the actual per-frame timestamps.
        # Need to reconstruct from DATE-OBS and EXPTIME. Since it's rolling-shutter, no readout time
        with fits.open(self.science_file_list[0]) as hdul:
            t1 = np.datetime64(hdul[0].header["DATAE-OBS"])
            dt = np.timedelta64(hdul[0].header["EXPTIME"], "ms")
        t0 = t1 - dt * len(self.science_file_list)
        startobs_array = np.arange(t0, t1, dt)
        if len(startobs_array) != len(self.science_file_list):
            # I haven't seen it happen, but there's an edge case here
            # Where the rounding of DATE-OBS causes issues.
            # In this case, we'll throw a warning and use linspace instead of
            # arange. This will result in slightly incorrect candences, but it shouldn't
            # matter too terribly much
            startobs_array = np.linspace(t0, t1, num=len(self.science_file_list))
            self.logger.warning(
                "Error in reconstructing timestamps! Cadence adjusted from:"
            )
            self.logger.warning(
                f"    {dt.astype(int)} ms to "
            )
            self.logger.warning(
                f"    {(startobs_array[1] - startobs_array[0]).astype(int)} ms"
            )


        for nfile, file in enumerate(tqdm.tqdm(
            self.science_file_list,
            desc="Running FRANCIS Calibration Loop",
            disable=not self.progress
        )):
            with fits.open(file) as hdul:
                startobs = startobs_array[nfile]
                exptime = int(hdul[0].header["EXPTIME"])
                endobs = startobs + np.timedelta64(exptime, "ms")
                data = np.mean(hdul[0].data, axis=0) - self.dark
            for i in range(data.shape[0]):
                data[i] = scind.shift(data[i], self.fiber_map_skews[i], mode="nearest")
            for j in range(data.shape[1]):
                data[:, j] = scind.shift(data[:, j], self.fiber_map_vskews[j], mode="nearest")
            for i in range(data.shape[0]):
                data[i] = scind.shift(data[i], self.rigid_xskews[i], mode="nearest")
            for j in range(data.shape[1]):
                data[:, j] = scind.shift(data[:, j], self.rigid_yskews[j], mode="nearest")
            data /= self.gain_table
            data /= self.spectral_gain
            # Reshape into 3D
            data_mapped = np.zeros((
                self.fiber_map_2d_tweaked.shape[1],
                self.fiber_map_2d_tweaked.shape[2],
                data.shape[1]
            )) # Y, X, lambda
            for y in range(data_mapped.shape[0]):
                for x in range(data_mapped.shape[1]):
                    # Clip the edge fibers
                    fiber_yrange = self.fiber_map_2d_tweaked[1:-1, y, x].astype(int)
                    # Sum the middle 3 fibers
                    if not all(fiber_yrange == 0):
                        data_mapped[y, x, :] = np.sum(data[fiber_yrange, :], axis=0)
                    else:
                        data_mapped[y, x, :] = 0

            # Choose lines for analysis
            line_cores = []
            if nfile == 0 and self.analysis_ranges is None and self.spectral_analysis:
                mean_profile = np.mean(data_mapped, axis=(0, 1))
                print("Select spectral ranges (xmin, xmax) for overview maps. Close window when done.")
                # Approximate indices of line cores
                coarse_indices = spex.select_spans_singlepanel(
                    mean_profile, xarr=self.wavelength_array, fig_name="Select Lines for Analysis"
                )
                # Location of minimum in the range
                min_indices = [
                    spex.find_nearest(
                        mean_profile[coarse_indices[x][0]:coarse_indices[x][1]],
                        mean_profile[coarse_indices[x][0]:coarse_indices[x][1]].min()
                    ) + coarse_indices[x][0] for x in range(coarse_indices.shape[0])
                ]
                # Location of exact line core
                line_cores = [
                    spex.find_line_core(mean_profile[x - 5:x + 7]) + x - 5 for x in min_indices
                ]
                # Find start and end indices that put line cores at the center of the window.
                self.analysis_ranges = np.zeros(coarse_indices.shape)
                for j in range(coarse_indices.shape[0]):
                    average_delta = np.mean(np.abs(coarse_indices[j, :] - line_cores[j]))
                    self.analysis_ranges[j, 0] = int(round(line_cores[j] - average_delta, 0))
                    self.analysis_ranges[j, 1] = int(round(line_cores[j] + average_delta, 0) + 1)
            elif nfile == 0 and self.analysis_ranges is not None and self.spectral_analysis:
                mean_profile = np.nanmean(data_mapped, axis=(0, 1))
                self.analysis_ranges = self.analysis_ranges.astype(int)
                line_cores = [
                    spex.find_line_core(
                        mean_profile[self.analysis_ranges[x, 0]:self.analysis_ranges[x, 1]]
                    ) + self.analysis_ranges[x, 0] for x in range(self.analysis_ranges.shape[0])
                ]

            if self.plot:
                plt.ion()
                # Set up overview plots to blit data into.
                # Unlike SPINOR/FIRS, we'll do up to 2 maps.
                # 1.) a wide plot that overlays every spectrum, coded by row
                # 2.) a 2d imshow of the core of the selected line
                if len(line_cores) > 0:
                    field_im = data_mapped[:, :, int(round(line_cores[0]))]
                    field_title = "Line Core of Selection 1"
                else:
                    field_im = np.nanmean(data_mapped, axis=-1)
                    field_title = "Mean Intensity"
                if nfile == 0:
                    plot_params = self.set_up_live_plot(data_mapped, field_im, field_title)
                self.update_live_plot(
                    plot_params, data_mapped, field_im
                )
            self.write_reduced_spectral_file(
                data_mapped,
                startobs,
                endobs,
                exptime,
                nfile
            )
            print_average_image_progress()
        return

    def write_reduced_spectral_file(
            self,
            datacube: np.ndarray,
            startobs: np.datetime64, endobs:np.datetime64,
            exptime: int,
            nfile: int
        ) -> None:
        """Write FITS file with Level-1 FRANCIS data

        Parameters
        ----------
        datacube : np.ndarray
            Remapped fiber-bundle spectra, of shape (20, 20, 2048)
        startobs : np.timedelta64
            Start of observation
        endobs : np.timedelta64
            End of observation
        exptime : int
            Exposure time
        nfile : int
            Number of file in sequence
        """

        prsteps = [
            "DARK-SUBTRACTION",
            "DESKEWING",
            "WAVELENGTH-CALIBRATION",
            "FLATFIELDING",
            "FIBER-EXTRACTION",
        ]
        prstep_comments = [
            "francisCal/SSOSoft",
            "francisCal/SSOSoft",
            "FTS Atlas",
            "francisCal/SSOSoft",
            self.fiber_map_filename
        ]
        if self.dcss_params != {}:
            dcss_startidx = spex.find_nearest(self.dcss_params["TIME"], startobs)
            dcss_endidx =spex.find_nearest(self.dcss_params["TIME"], endobs)
            rotan = self.dcss_params["GDRAN"][dcss_startidx] - 13.3
            srad = self.dcss_params["SDIM"][dcss_startidx] / 2
            llvl = np.mean(self.dcss_params["LLVL"][dcss_startidx:dcss_endidx + 1])
            scin = np.mean(self.dcss_params["SCIN"][dcss_startidx:dcss_endidx + 1])
            slat = np.mean(self.dcss_params["SLAT"][dcss_startidx:dcss_endidx + 1])
            slng = np.mean(self.dcss_params["SLNG"][dcss_startidx:dcss_endidx + 1])
            center_coord = SkyCoord(
                slng * u.deg, slat * u.deg,
                obstime=startobs.astype(str), observer="earth",
                frame=frames.HeliographicStonyhurst
            ).transform_to(frames.Helioprojective)
            solarx = center_coord.Tx.value
            solary = center_coord.Ty.value
        else:
            rotan = srad = llvl = scin = solarx = solary = "N/A"
        ext0 = fits.PrimaryHDU()
        ext0.header["DATE"] = (np.datetime64("now").astype(str), "File Creation Date and Time (UTC)")
        ext0.header["ORIGIN"] = "NMSU/SSOC"
        ext0.header["TELESCOP"] = ("DST", "Dunn Solar Telescope, Sacramento Peak NM")
        ext0.header["INSTRUME"] = ("FRANCIS", "Jess et. al., 2022")
        ext0.header["AUTHOR"] = "sellers"
        ext0.header["CAMERA"] = "KURO-2k"
        ext0.header["DATA_LEV"] = 1
        ext0.header["WAVEBAND"] = self.wavelength
        ext0.header["STARTOBS"] = startobs.astype(str)
        ext0.header["DATE-OBS"] = startobs.astype(str)
        ext0.header["ENDOBS"] = endobs.astype(str)
        ext0.header["DATE-END"] = endobs.astype(str)
        ext0.header["BTYPE"] = "Intensity"
        ext0.header["BUNIT"] = "Corrected DN"
        ext0.header["EXPTIME"] = (exptime, "ms for single exposure")
        ext0.header["SLIT-WID"] = (50, "[um] FRANCIS Slit Width")
        ext0.header["FIB-APT"] = (self.FIBER_APERTURE, "um, Diameter of fiber head")
        ext0.header["FIB-ASEC"] = (round(self.FIBER_APERTURE_ASEC, 3), "asec, Diameter of fiber head")
        ext0.header["FIB-DIAM"] = (self.FIBER_WIDTH, "um, Diameter of fiber+clad")
        ext0.header["FIB-XU"] = (self.FIBER_WIDTH, "um, Horizontal fiber spacing")
        ext0.header["FIB-XA"] = (round(self.HSPACING_ASEC, 3), "asec, Horizontal fiber spacing")
        ext0.header["FIB-YU"] = (round(self.VSPACING, 3), "um, Vertical fiber spacing")
        ext0.header["FIB-YA"] = (round(self.VSPACING_ASEC, 3), "asec, Vertical fiber spacing")
        ext0.header["WAVEUNIT"] = (-10, "10^(WAVEUNIT), Angstrom")
        ext0.header["WAVEREF"] = ("FTS", "Kurucz 1984 Atlas Used in Wavelength Determination")
        ext0.header["WAVEMIN"] = (round(self.wavelength_array[0], 3), "[AA] Angstrom")
        ext0.header["WAVEMAX"] = (round(self.wavelength_array[-1], 3), "[AA], Angstrom")
        ext0.header["GRPERMM"] = (self.grating_lpmm, "[mm^-1] Lines per mm of Grating")
        ext0.header["SPORDER"] = (1, "Spectral Order")
        if rotan != "N/A":
            ext0.header["RSUN_ARC"] = (srad, "asec, photosphere radius")
            ext0.header["XCEN"] = (round(solarx, 3), "[arcsec], Solar-X of Map Center")
            ext0.header["YCEN"] = (round(solary, 3), "[arcsec], Solar-Y of Map Center")
            ext0.header["ROT"] = (round(rotan, 3), "[degrees], Rotation from Solar-North")
            ext0.header["SCIN"] = (scin, "Average scintillation during obs.")
            ext0.header["LLVL"] = (llvl, "Average light level during obs.")
        ext0.header["FOVX"] = (
            round(datacube.shape[1] * self.HSPACING_ASEC, 3),
            "[arcsec], Field-of-view of fiber bundle horiz."
        )
        ext0.header["FOVY"] = (
            round(datacube.shape[0] * self.VSPACING_ASEC, 3),
            "[arcsec], Field-of-view of fiber bundle vert."
        )
        for i in range(len(prsteps)):
            ext0.header["PRSTEP" + str(int(i + 1))] = (prsteps[i], prstep_comments[i])
        ext0.header["COMMENT"] = "Full WCS Information Contained in Individual Data HDUs"
        ext0.header.insert(
            "DATA_LEV",
            ("", "======== DATA SUMMARY ========"),
            after=True
        )
        ext0.header.insert(
            "WAVEUNIT",
            ("", "======== SPECTROGRAPH CONFIGURATION ========")
        )
        if rotan != "N/A":
            ext0.header.insert(
                "RSUN_ARC",
                ("", "======== POINTING INFORMATION ========")
            )
        ext0.header.insert(
            "PRSTEP1",
            ("", "======== CALIBRATION PROCEDURE OUTLINE ========")
        )

        ext = fits.ImageHDU(datacube)
        ext.header["EXTNAME"] = "FIBERS"
        ext.header["DATE-OBS"] = startobs.astype(str)
        ext.header["DATE-END"] = endobs.astype(str)
        if rotan != "N/A":
            ext.header["RSUN_ARC"] = srad
            ext.header["CDELT1"] = (round(self.HSPACING_ASEC, 3), "arcsec")
            ext.header["CDELT2"] = (round(self.VSPACING_ASEC, 3), "arcsec")
            ext.header["CDELT3"] = (self.wavelength_array[1] - self.wavelength_array[0], "Angstrom")
            ext.header["CTYPE1"] = "HPLN-TAN"
            ext.header["CTYPE2"] = "HPLT-TAN"
            ext.header["CTYPE3"] = "WAVE"
            ext.header["CUNIT1"] = "arcsec"
            ext.header["CUNIT2"] = "arcsec"
            ext.header["CUNIT3"] = "Angstrom"
            ext.header["CRVAL1"] = (solarx, "Solar-X, arcsec")
            ext.header["CRVAL2"] = (solary, "Solar-Y, arcsec")
            ext.header["CRVAL3"] = (self.wavelength_array[0], "Angstrom")
            ext.header["CRPIX1"] = np.mean(np.arange(datacube.shape[0])) + 1
            ext.header["CRPIX2"] = np.mean(np.arange(datacube.shape[1])) + 1
            ext.header["CRPIX3"] = 1
            ext.header["CROTA2"] = (round(rotan, 3), "degrees")

        ext_wvl = fits.ImageHDU(self.wavelength_array)
        ext_wvl.header["EXTNAME"] = "lambda-coordinate"
        ext_wvl.header["BTYPE"] = "lambda axis"
        ext_wvl.header["BUNIT"] = "[AA]"

        fits_hdus = [ext0, ext, ext_wvl]
        outname = self.reduced_file_pattern.format(
            self.date,
            self.time,
            self.wavelength,
            nfile
        )
        outfile = os.path.join(self.calibrated_base, outname)
        fits_hdulist = fits.HDUList(fits_hdus)
        fits_hdulist.writeto(outfile, overwrite=True)

        return

    def set_up_live_plot(
            self, datamap: np.ndarray, fieldim: np.ndarray, fieldtitle: str
        ) -> tuple[matplotlib.figure.Figure, matplotlib.image.AxesImage, list]:
        """Sets up live plotting

        Parameters
        ----------
        datamap : np.ndarray
            2D ferrule map (ny, nx, nlambda)
        fieldim : np.ndarray
            Map of either line core intensity or mean intensity
        fieldtitle : str
            Description of fieldim
        """
        # Close all figures to reset plotting
        plt.close("all")

        # Required for live plotting
        plt.ion()
        plt.pause(0.005)

        # Set up spectral data. Large window, 20 plots (1 per row w/ 20 lines)
        spec_fig = plt.figure("Reduced Spectra", figsize=(15, 10))
        spectra_masterlist = []
        for i in range(datamap.shape[0]):
            ax = spec_fig.add_subplot(5, 4, i + 1)
            spectra_list = []
            for j in range(datamap.shape[1]):
                # Plots go top to bottom, ferrule goes bottom to top
                if not all(datamap[datamap.shape[0] - i - 1, j, :] == 0):
                    line, = ax.plot(self.wavelength_array, datamap[datamap.shape[0] - i - 1, j, :])
                    spectra_list.append(line)
            spectra_masterlist.append(spectra_list)
            ax.set_title(f"Ferrule row {datamap.shape[0] - i}")
        spec_fig.tight_layout()

        field_fig = plt.figure(fieldtitle, figsize=(5, 5))
        field_ax = field_fig.add_subplot(111)
        field_ax.set_title(fieldtitle)
        field_ax.set_xlabel("Ferrule Columns")
        field_ax.set_ylabel("Ferrule Rows")
        field = field_ax.imshow(
            fieldim, origin="lower", cmap="gray",
            vmin=np.nanmean(fieldim) - 3 * np.nanstd(fieldim),
            vmax=np.nanmean(fieldim) + 3 * np.nanstd(fieldim)
        )

        plt.show(block=False)
        plt.pause(0.05)

        return spec_fig, spectra_masterlist, field_fig, field

    def update_live_plot(
            self, plot_params: tuple, datamap: np.ndarray, fieldim: np.ndarray
        ) -> None:
        """Updates live plot

        Parameters
        ----------
        plot_params : tuple
            Contains spectral fig, list of spectra, field_fig, field image
        datamap : np.ndarray
            Reduced data
        fieldim : np.ndarray
            Image of field
        """
        for i in range(datamap.shape[0]):
            for j in range(datamap.shape[1]):
                ctr = 0 # Since it's a list w/ appended lines, will not have the same shape as the fibermap
                if not all(datamap[datamap.shape[0] - i - 1, j, :] == 0):
                    plot_params[1][i][ctr].set_ydata(datamap[datamap.shape[0] - i - 1, j, :])
                    ctr += 1
        plot_params[0].canvas.draw()
        plot_params[0].canvas.flush_events()
        plot_params[3].set_array(fieldim)
        plot_params[3].set_norm(
            matplotlib.colors.Normalize(
                vmin=np.nanmean(fieldim) - 3 * np.nanstd(fieldim),
                vmax=np.nanmean(fieldim) + 3 * np.nanstd(fieldim)
            )
        )
        plot_params[2].canvas.draw()
        plot_params[2].canvas.flush_events()
        return

    def francis_get_polcal(self) -> None:
        pass

    def reduce_francis_polarimetry_maps(self) -> None:
        pass

    def parse_dcss(self) -> None:
        """
        Parser for DCSS/VTT logs to set up time, scintillation, light level, and pointing arrays.
        """
        # No DCSS Log Set
        if self.dcss_log == "":
            return
        with open(self.dcss_log, "r") as f:
            dcss_lines = f.readlines()

        lat = []
        lon = []
        scin = []
        timestamp = []
        llvl = []
        gdran = []
        srad = []
        for line in dcss_lines:
            if "DST_TIME" in line:
                timestamp.append(
                    np.datetime64(
                        line.split("=")[1].split("/")[0].strip().replace("\'","")
                    )
                )
            # Reconstructing ICC Datetimes, which are kept in "DATE" and "TIME" lines
            # DCSS has WP_TIME, RM1_TIME, PT4_TIME, etc... we want to make sure we're not
            # getting these lines.  Since ICC logs don't have the underscore character
            # (except in CTRK TRACK_ERROR), we can filter by "_".
            # This is a blatant hack, but it works.
            elif ("TIME" in line) & ("_" not in line):
                timestamp.append(line.split(" ")[-1].replace("\n", ""))
            elif "DATE" in line:
                date = line.replace("\n", "").split(" ")[-1].split("/")
                year = "20" + date[-1]
                month = date[0]
                day = date[1]
                fdate = "-".join([year, month, day])
                timestamp[-1] = fdate + "T" + timestamp[-1]

            if "DST_SLAT" in line:
                lat.append(
                    float(
                        line.split("=")[1].split("/")[0].strip()
                    )
                )
            elif "VTT SLAT" in line:
                lat.append(
                    float(
                        line.split(" ")[-1].replace("\n", "")
                    )
                )
            if "DST_SLNG" in line:
                lon.append(
                    float(
                        line.split("=")[1].split("/")[0].strip()
                    )
                )
            elif "VTT SLNG" in line:
                lon.append(
                    float(
                        line.split(" ")[-1].replace("\n", "")
                    )
                )
            if "DST_SEE" in line:
                scin.append(
                    float(
                        line.split("=")[1].split("/")[0].strip()
                    )
                )
            elif "VTT SEE" in line:
                scin.append(
                    float(
                        line.split(" ")[-1].replace("\n", "")
                    )
                )
            if "DST_LLVL" in line:
                llvl.append(
                    float(
                        line.split("=")[1].split("/")[0].strip()
                    )
                )
            elif "VTT LLVL" in line:
                llvl.append(
                    float(
                        line.split(" ")[-1].replace("\n", "")
                    )
                )
            if "DST_GDRN" in line:
                gdran.append(
                    float(
                        line.split("=")[1].split("/")[0].strip()
                    )
                )
            elif "VTT GDRN" in line:
                gdran.append(
                    float(
                        line.split(" ")[-1].replace("\n" ,"")
                    )
                )
            if "DST_SDIM" in line:
                srad.append(
                    float(
                        line.split("=")[1].split("/")[0].strip()
                    )
                )
            elif "VTT SDIM" in line:
                srad.append(
                    float(
                        line.split(" ")[-1].replace("\n", "")
                    )
                )

        self.dcss_params = {
            "TIME": np.array(timestamp, dtype="datetime64[ms]"),
            "SLAT": np.array(lat),
            "SLNG": np.array(lon),
            "SCIN": np.array(scin),
            "LLVL": np.array(llvl),
            "SDIM": np.array(srad),
            "GDRAN": np.array(gdran)
        }
        return
