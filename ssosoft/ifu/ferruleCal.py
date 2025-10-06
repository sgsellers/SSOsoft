import configparser
import glob
import logging
import os
from importlib import resources

import astropy.io.fits as fits
import astropy.units as u
import numpy as np
import scipy.ndimage as scind
import scipy.signal as scig
import sunpy.map as smap
import tqdm
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames

from ssosoft.tools import alignment_tools as align


def _find_nearest(array, value):
    """Finds index of closest value in array"""
    return np.abs(array-value).argmin()

def _mfgs(image: np.ndarray, kernel_size:int=3) -> float:
    """
    Median filter gradiant similarity from Deng et. al., 2015.
    Used for tracking quality of image
    """
    med_img = scind.median_filter(image, size=kernel_size)
    grad_img = np.gradient(image)

    grad = (np.sum(np.abs(grad_img[0][image != 0])) + np.sum(np.abs(grad_img[1][image != 0])))
    med_grad = np.sum(np.abs(np.gradient(med_img)))

    mfgs_metric = ((2 * med_grad * grad) /
                   (med_grad**2 + grad**2))

    return mfgs_metric


class FerruleCal():
    """
    The Sunspot Solar Observatory Consotrium's software for reducing FRANCIS
    (The Fibre-Resolved opticAl and Near-ultraviolet Czerny-turner Imaging Spectropolarimeter)
    ferrule context imaging data from the Dunn Solar Telescope.
    """
    from ssosoft import ssosoftConfig
    IMAGESHAPE=(1002, 1004) # Default ROSA chip size
    DST_PLATE_SCALE=206264.806 / 54864. # arcsec per radian / primary f.l.
    def __init__(self, config_file: str) -> None:
        """Initialize FerruleCal and FrancisCal parent class

        Parameters
        ----------
        config_file : str
            Path to configuration file
        """

        self.config_file = config_file
        self.ferrule_data_base = ""
        self.ferrule_dark_base = ""
        self.ferrule_flat_base = ""
        self.ferrule_target_base = ""
        self.ferrule_linegrid_base = ""

        self.ferrule_work_base = ""
        self.ferrule_calibrated_base = ""

        self.ferrule_data_file_pattern = "*"
        self.ferrule_dark_file_pattern = "*"
        self.ferrule_flat_file_pattern = "*"
        self.ferrule_target_file_pattern = "*"
        self.ferrule_linegrid_file_pattern = "*"

        self.ferrule_obsdate = ""
        self.ferrule_obstime = ""
        self.correct_time = False
        self.ferrule_xscale = 0.
        self.ferrule_yscale = 0.
        self.ferrule_plate_scale = []

        self.correct_plate_scale = False
        self.create_fiducial_maps = True
        self.channel_translation = []

        self.camera_rotation = True
        self.progress = False

        self.dcss_log = ""
        self.dcss_times = np.array([])
        self.dcss_lat = np.array([])
        self.dcss_lon = np.array([])
        self.dcss_see = np.array([])
        self.dcss_llvl = np.array([])
        self.dcss_gdran = np.array([])
        self.dcss_sdim = np.array([])

        self.ferrule_dark_file = ""
        self.ferrule_flat_file = ""
        self.ferrule_gain_file = ""
        self.ferrule_linegrid_file = ""
        self.ferrule_target_file = ""

        self.avg_dark = np.empty(0)
        self.avg_flat = np.empty(0)
        self.gain = np.empty(0)
        self.avg_lgrd = np.empty(0)
        self.avg_targ = np.empty(0)

        self.ferrule_position = []

        self.burst_type = "registration" # or "continuous" if run for entire sequence
        self.francis_setup = "default"


        """See S. Sellers setup notes for FRANCIS for a summary of the "default"
        setup mode. The default mode is meant for use with every other DST
        facility instrument. The FRANCIS beam comes from the facility AO,
        reflects off the Lumonics Visible Reflector beamsplitter (380-700 nm),
        then again off a panchromatic 50/50 beam splitter, through a 225 mm lens,
        a 260 mm lens, and a 300 mm lens to the fiber ferrule. The back reflection
        of the ferrule takes the same path through the lenses, through the 50/50 again,
        and is imaged by a 500 mm lens onto a ROSA camera.
        (with some pretty bad astigmatism)

        If you are still using this setup, the code assumes that the reference image
        of the ferrule does not need to be interpolated onto a new grid, and allows
        a better alignment using the plate scale included in that file.

        If something has been altered in the setup, the alignment will include a guess
        at interpolation using the user-provided scale.
        """

        return

    def ferrule_run_calibration(self) -> None:
        """Function block for cal loop"""
        self.ferrule_configure_run()
        self.ferrule_get_cal_images()
        self.ferrule_save_cal_images()
        self.parse_dcss()
        self.ferrule_save_processed_maps()
        return

    def ferrule_configure_run(self) -> None:
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
        self.ferrule_data_base = config["FERRULE"]["dataBase"]
        # If these are undefined, assign them to the same directory as dataBase
        # Useful for ROSA where all files are in the same directory.
        self.ferrule_dark_base = config["FERRULE"]["darkBase"] if "darkbase" \
            in config["FERRULE"].keys() else self.ferrule_data_base
        self.ferrule_flat_base = config["FERRULE"]["flatBase"] if "flatbase" \
            in config["FERRULE"].keys() else self.ferrule_data_base
        self.ferrule_target_base = config["FERRULE"]["targetBase"] if "targetbase" \
            in config["FERRULE"].keys() else self.ferrule_data_base
        self.ferrule_linegrid_base = config["FERRULE"]["linegridBase"] if "linegridbase" \
            in config["FERRULE"].keys() else self.ferrule_data_base

        self.ferrule_work_base = config["FERRULE"]["workBase"]
        if not os.path.isdir(self.ferrule_work_base):
            print("{0}: os.mkdir: attempting to create directory:"
                    "{1}".format(__name__, self.ferrule_work_base)
                    )
            try:
                os.mkdir(self.ferrule_work_base)
            except Exception as err:
                print("An exception was raised: {0}".format(err))
                raise

        # If everything is in its own directory, having each in the config file is silly
        self.ferrule_data_file_pattern = config["FERRULE"]["dataFilePattern"] if \
            "datafilepattern" in config["FERRULE"].keys() else self.ferrule_data_file_pattern
        self.ferrule_dark_file_pattern = config["FERRULE"]["darkFilePattern"] if \
            "darkfilepattern" in config["FERRULE"].keys() else self.ferrule_data_file_pattern
        self.ferrule_flat_file_pattern = config["FERRULE"]["flatFilePattern"] if \
            "flatfilepattern" in config["FERRULE"].keys() else self.ferrule_data_file_pattern
        self.ferrule_target_file_pattern = config["FERRULE"]["targetFilePattern"] if \
            "targetfilepattern" in config["FERRULE"].keys() else self.ferrule_data_file_pattern
        self.ferrule_linegrid_file_pattern = config["FERRULE"]["linegridFilePattern"] if \
            "linegridfilepattern" in config["FERRULE"].keys() else self.ferrule_linegrid_file_pattern

        self.ferrule_obsdate = config["FERRULE"]["obsDate"]
        self.ferrule_obstime = config["FERRULE"]["obsTime"]

        self.correct_time = config["FERRULE"]["correctTime"] if "correcttime" in \
            config["FERRULE"].keys() else self.correct_time
        self.correct_time = True if str(self.correct_time).lower() == "true" else False

        self.ferrule_xscale = float(config["FERRULE"]["scaleX"])
        self.ferrule_yscale = float(config["FERRULE"]["scaleY"])
        self.ferrule_plate_scale = [self.ferrule_xscale, self.ferrule_yscale]

        self.correct_plate_scale = config["FERRULE"]["correctPlateScale"] if "correctplatescale" in \
            config["FERRULE"].keys() else self.correct_plate_scale
        self.correct_plate_scale = True if str(self.correct_plate_scale).lower() == "true" else False

        self.create_fiducial_maps = config["FERRULE"]["createFiducialMaps"] if "createfiducialmaps" in \
            config["FERRULE"].keys() else self.create_fiducial_maps
        self.create_fiducial_maps = True if str(self.create_fiducial_maps).lower() == "true" else False

        if "channeltranslation" in config["FERRULE"].keys():
            trans_string = config["FERRULE"]["channelTranslation"]
            if trans_string != "" and trans_string.lower() != "none":
                self.channel_translation = trans_string.split(",")

        self.camera_rotation = config["FERRULE"]["cameraRotation"] if "camerarotation" in \
            config["FERRULE"].keys() else self.camera_rotation
        self.camera_rotation = True if str(self.camera_rotation).lower() == "true" else False

        self.progress = config["FERRULE"]["progress"] if "progress" in \
            config["FERRULE"].keys() else self.progress
        self.progress = True if str(self.progress).lower() == "true" else False

        self.ferrule_dark_file = os.path.join(self.ferrule_work_base, "FERRULE_DARK.fits")
        self.ferrule_flat_file = os.path.join(self.ferrule_work_base, "FERRULE_FLAT.fits")
        self.ferrule_gain_file = os.path.join(self.ferrule_work_base, "FERRULE_GAIN.fits")
        if self.create_fiducial_maps:
            self.ferrule_linegrid_file = os.path.join(
                self.ferrule_work_base, "FERRULE_LINEGRID.fits"
            )
            self.ferrule_target_file = os.path.join(
                self.ferrule_work_base, "FERRULE_TARGET.fits"
            )
        self.burst_type = config["FERRULE"]["acquisitionType"] if "acquisitiontype" in \
            config["FERRULE"].keys() else self.burst_type
        self.francis_setup = config["FERRULE"]["setupType"] if "setuptype" in \
            config["FERRULE"].keys() else self.francis_setup

        self.ferrule_calibrated_base = os.path.join(self.ferrule_work_base, "ferrule_images")
        if not os.path.isdir(self.ferrule_calibrated_base):
            print("{0}: os.mkdir: attempting to create directory:"
                    "{1}".format(__name__, self.ferrule_calibrated_base)
                    )
            try:
                os.mkdir(self.ferrule_calibrated_base)
            except Exception as err:
                print("An exception was raised: {0}".format(err))
                raise


        self.logfile = os.path.join(self.ferrule_work_base, f"{self.ferrule_obstime}_FRANCIS_FERRULE.log")
        logging.config.fileConfig(self.config_file, defaults={"logfilename": self.logfile})
        self.logger = logging.getLogger("FRANCIS_FERRULE_Log")
        # Intro message
        self.logger.info(f"This is SSOsoft version {self.ssosoftConfig.__version__}")
        self.logger.info(f"Contact {self.ssosoftConfig.__email__} to report bugs, make suggestions, "
                         "of contribute")
        self.logger.info("Now configuring this FRANCIS ferrule camera data calibration run.")

        return

    def rosa_average_image_from_list(self, filelist: list) -> np.ndarray:
        """ROSA-specific average image from list"""
        def rosa_print_average_image_progress():
            if not fnum % 100:
                self.logger.info("Progress: "
                                 "{:0.1%}.".format(fnum / num_img)
                                 )

        self.logger.info("Computing average image from {0} files "
                         "in directory: {1}".format(
            len(filelist), os.path.dirname(filelist[0])
        ))

        avg_im = np.zeros(self.IMAGESHAPE, dtype=np.float32)
        fnum = 0
        num_img = len(filelist)
        for file in tqdm.tqdm(
            filelist,
            desc=f"Computing Average Image from {len(filelist)} files",
            disable=not self.progress
        ):
            with fits.open(file) as hdul:
                if fnum == 0:
                    num_img = len(filelist) * len(hdul[1:])
                for hdu in hdul[1:]:
                    avg_im += hdu.data
                    fnum += 1
                    rosa_print_average_image_progress()
        avg_im /= fnum
        self.logger.info("Images averaged/images predicted: "
                         "{0}/{1}".format(fnum, num_img)
                         )
        try:
            assert fnum == num_img
        except AssertionError:
            self.logger.warning(
                "Number of images averaged does not match the "
                "number predicted. If instrument is ROSA, then "
                "this might be OK. Otherwise, something went wrong."
            )
            self.logger.warning("This calibration will continue to run.")

        self.logger.info("Average complete, directory: "
                         "{0}".format(os.path.dirname(filelist[0]))
                         )
        return avg_im

    def translate_image(self, image:np.ndarray) -> np.ndarray:
        """Performs translations to an image

        Parameters
        ----------
        image : np.ndarray
            Untranslated image

        Returns
        -------
        image: np.ndarray
            Translated image
        """
        if self.camera_rotation:
            image = np.rot90(image)
        for translation in self.channel_translation:
            if translation.lower() == "rot90":
                image = np.rot90(image)
            elif translation.lower() == "flip":
                image = np.flip(image)
            elif translation.lower() == "fliplr":
                image = np.fliplr(image)
            elif translation.lower() == "flipud":
                image = np.flipud(image)

        return image

    def ferrule_get_cal_images(self):
        """Reads average dark, flat, and gain files, stores as class
        attributes if these exist. If these finles do not exist,
        create the average images and store.
        """
        if os.path.exists(self.ferrule_dark_file):
            self.logger.info(f"Average dark file found: {self.ferrule_dark_file}")
            self.logger.info("Reading average dark")
            with fits.open(self.ferrule_dark_file) as hdul:
                self.avg_dark = hdul[0].data
        else:
            darklist = glob.glob(
                os.path.join(self.ferrule_dark_base, self.ferrule_dark_file_pattern)
            )
            if len(darklist) == 0:
                self.logger.warning(
                    f"No dark files found in {self.ferrule_dark_base}!"
                )
                self.logger.warning(
                    "Proceeding with incomplete calibration!"
                )
                self.avg_dark = np.zeros(self.IMAGESHAPE)
            else:
                self.avg_dark = self.rosa_average_image_from_list(darklist)
        if os.path.exists(self.ferrule_flat_file):
            self.logger.info(f"Average flat file found: {self.ferrule_flat_file}")
            self.logger.info("Reading average flat.")
            with fits.open(self.ferrule_flat_file) as hdul:
                self.avg_flat = hdul[0].data
        else:
            flatlist = glob.glob(
                os.path.join(self.ferrule_flat_base, self.ferrule_flat_file_pattern)
            )
            if len(flatlist) == 0:
                self.logger.warning(
                    f"No flat files found in {self.ferrule_flat_base}"
                )
                self.logger.warning("Proceeding with incomplete calibration!")
                self.logger.warning(
                    "Ferrule position cannot be accurately determined without flat field"
                )
                self.avg_flat = np.ones(self.IMAGESHAPE)
            else:
                self.avg_flat = self.rosa_average_image_from_list(flatlist)

        if os.path.exists(self.ferrule_gain_file):
            self.logger.info(f"Gain file found: {self.ferrule_gain_file}")
            self.logger.info("Reading gain file.")
            with fits.open(self.ferrule_gain_file) as hdul:
                self.gain = hdul[0].data
                self.ferrule_position = [hdul[0].header["FERR-X"], hdul[0].header["FERR-Y"]]
        elif not (self.avg_flat == 1).all():
            self.ferrule_create_gain()
        else:
            self.gain = self.translate_image(np.ones(self.IMAGESHAPE))

        if os.path.exists(self.ferrule_linegrid_file) and self.create_fiducial_maps:
            self.logger.info(f"Linegrid file found: {self.ferrule_linegrid_file}")
            self.logger.info("Loading linegrid file.")
            with fits.open(self.ferrule_linegrid_file) as hdul:
                self.avg_lgrd = hdul[0].data
                plate_scale = os.path.join(self.ferrule_work_base, "ferrule_plate_scale.txt")
            if self.correct_plate_scale and os.path.exists(plate_scale):
                self.ferrule_plate_scale = np.loadtxt(plate_scale)
                self.logger.info("Updating plate scale.")
                self.logger.info(f"X-scale (calc): {self.ferrule_xscale}")
                self.logger.info(f"X-scale (meas): {self.ferrule_plate_scale[0]}")
                self.logger.info(f"Y-scale (calc): {self.ferrule_yscale}")
                self.logger.info(f"Y-scale (meas): {self.ferrule_plate_scale[1]}")
            elif self.correct_plate_scale:
                self.determine_grid_spacing()
        elif self.create_fiducial_maps:
            lgrdlist = glob.glob(os.path.join(
                self.ferrule_linegrid_base, self.ferrule_linegrid_file_pattern
            ))
            if len(lgrdlist) == 0:
                self.logger.warning(
                    f"No linegrid images found in {self.ferrule_linegrid_base}"
                )
                self.logger.warning(
                    f"With file pattern: {self.ferrule_linegrid_file_pattern}"
                )
                self.logger.info("Plate scale cannot be updated")
            else:
                self.avg_lgrd = self.rosa_average_image_from_list(lgrdlist)
                self.avg_lgrd = self.translate_image(self.avg_lgrd - self.avg_dark) / self.gain
                if self.correct_plate_scale:
                    self.determine_grid_spacing()
        else:
            self.logger.info("Skipping linegrid creation. Plate scale has not been updated.")
        if os.path.exists(self.ferrule_target_file) and self.create_fiducial_maps:
            self.logger.info(f"Target file found: {self.ferrule_target_file}")
            self.logger.info("Loading target file.")
            with fits.open(self.ferrule_target_file) as hdul:
                self.avg_targ = hdul[0].data
        elif self.create_fiducial_maps:
            targlist = glob.glob(os.path.join(
                self.ferrule_target_base, self.ferrule_target_file_pattern
            ))
            if len(targlist) == 0:
                self.logger.warning(
                    f"No target images found in {self.ferrule_target_base}"
                )
                self.logger.warning(
                    f"With file pattern: {self.ferrule_target_file_pattern}"
                )
                self.logger.warning(
                    "With no target image, rigid alignment to a reference channel cannot be performed"
                )
            else:
                self.avg_targ = self.rosa_average_image_from_list(targlist)
                self.avg_targ = self.translate_image(self.avg_targ - self.avg_dark) / self.gain
        else:
            self.logger.info(
                "Skipping target creation. Rigid alignment to reference will not be done."
            )
        if (self.gain == 1).all():
            # No cal images for the day. User-select fiber head position.
            flist = sorted(glob.glob(os.path.join(self.ferrule_data_base, self.ferrule_data_file_pattern)))
            with fits.open(flist[0]) as hdul:
                img = self.translate_image(hdul[1].data)
            boundary = align.click_and_drag_select(
                img,
                figtext="Click and drag to select the fiber bundle!"
            )
            self.ferrule_position = [np.mean(boundary[2:]), np.mean(boundary[:2])]
        return

    def ferrule_save_cal_images(self) -> None:
        """
        Saves average dark, flat, gain, linegrid, and target images
        """
        if os.path.exists(self.ferrule_dark_file):
            self.logger.info(f"Dark file already exists: {self.ferrule_dark_file}")
        else:
            self.logger.info(
                f"Saving average dark: {self.ferrule_dark_file}"
            )
            self.ferrule_save_fits_image(self.avg_dark, self.ferrule_dark_file)
        if os.path.exists(self.ferrule_flat_file):
            self.logger.info(f"Flat file already exists: {self.ferrule_flat_file}")
        else:
            self.logger.info(
                f"Saving average flat: {self.ferrule_flat_file}"
            )
            self.ferrule_save_fits_image(self.avg_flat, self.ferrule_flat_file)
        if os.path.exists(self.ferrule_gain_file):
            self.logger.info(f"Gain file already exists: {self.ferrule_gain_file}")
        else:
            self.logger.info(
                f"Saving gain: {self.ferrule_gain_file}"
            )
            self.ferrule_save_fits_image(
                self.gain,
                self.ferrule_gain_file,
                header={"FERR-X":self.ferrule_position[0], "FERR-Y":self.ferrule_position[1]}
            )
        if self.avg_targ.shape != np.empty(0).shape:
            if os.path.exists(self.ferrule_target_file):
                self.logger.info(
                    f"Target file already exists: {self.ferrule_target_file}"
                )
            else:
                self.logger.info(
                    f"Saving average target: {self.ferrule_target_file}"
                )
                # Well this is tortured
                obsdate = "-".join(
                    [self.ferrule_obsdate[:4], self.ferrule_obsdate[4:6], self.ferrule_obsdate[6:]]
                )
                obstime = ":".join(
                    [self.ferrule_obstime[:2], self.ferrule_obstime[2:4], self.ferrule_obstime[4:]]
                )
                # For channel co-alignment, we need minimal WCS, mostly for the plate scale
                target_pointing_info = {
                    "STARTOBS": obsdate + "T" + obstime,
                    "CRVAL1": 0.0,
                    "CRVAL2": 0.0,
                    "CTYPE1": "HPLN-TAN",
                    "CTYPE2": "HPLT-TAN",
                    "CUNIT1": "arcsec",
                    "CUNIT2": "arcsec",
                    "CDELT1": self.ferrule_plate_scale[0],
                    "CDELT2": self.ferrule_plate_scale[1],
                    "CRPIX1": self.avg_targ.shape[1] / 2,
                    "CRPIX2": self.avg_targ.shape[0] / 2,
                    "CROTA2": 0.0
                }
                self.ferrule_save_fits_image(
                    self.avg_targ,
                    self.ferrule_target_file,
                    header=target_pointing_info
                )
        if self.avg_lgrd.shape != np.empty(0).shape:
            if os.path.exists(self.ferrule_linegrid_file):
                self.logger.info(
                    f"Linegrid file already exists: {self.ferrule_linegrid_file}"
                )
            else:
                self.logger.info(
                    f"Saving average linegrid: {self.ferrule_linegrid_file}"
                )
                self.ferrule_save_fits_image(
                    self.avg_lgrd,
                    self.ferrule_linegrid_file
                )
        return

    def determine_grid_spacing(self):
        """Determines plate scale from grid images. Since the ferrule image is afflicted by some 
        pretty bad astigmatism and also has a great bit fiber head image in it, we've got to do a
        couple extra steps compared to the ROSA/HARDcam version of this function.
        For one, we'll use a 256-pixel window that's opposite the ferrule.
        For another, we'll smooth the profiles to try and minimize the astigmatism.
        """
        linerange = align.click_and_drag_select(
            self.avg_lgrd,
            figtext="Click and drag to select a region for grid scale determination"
        )
        subimage = self.avg_lgrd[linerange[0]:linerange[1], linerange[2]:linerange[3]]
        xmedian_spacing = []
        for i in range(subimage.shape[0]):
            profile = 1/self.avg_lgrd[i, :]
            profile /= profile.max()
            profile = scind.median_filter(profile, 7)
            peaks, _ = scig.find_peaks(profile, height=0.9)
            xmedian_spacing.append(np.nanmedian(peaks[1:] - peaks[:-1]))
        xpix_scale = self.DST_PLATE_SCALE / np.nanmedian(xmedian_spacing)
        if np.abs((xpix_scale - self.ferrule_xscale) / self.ferrule_xscale) < 0.1:
            # Within 10% of initial value, update
            self.ferrule_xscale = xpix_scale
        else:
            self.logger.warning(
                "Plate scale [x] could not be updated! Original Value: {0:03f}. Estimated Value: {1:03f}".format(
                    self.ferrule_xscale, xpix_scale
                )
            )

        ymedian_spacing = []
        for i in range(subimage.shape[1]):
            profile = 1/self.avg_lgrd[:, i]
            profile /= profile.max()
            profile = scind.median_filter(profile, 7)
            peaks, _ = scig.find_peaks(profile, height=0.9)
            ymedian_spacing.append(np.nanmedian(peaks[1:] - peaks[:-1]))

        ypix_scale = self.DST_PLATE_SCALE / np.nanmedian(ymedian_spacing)
        if np.abs((ypix_scale - self.ferrule_yscale) / self.ferrule_yscale) < 0.1:
            # Within 10% of initial value, update
            self.ferrule_yscale = ypix_scale
        else:
            self.logger.warning(
                "Plate scale [y] could not be updated! Original Value: {0:03f}. Estimated Value: {1:03f}".format(
                    self.ferrule_yscale, ypix_scale
                )
            )
        self.ferrule_plate_scale = [self.ferrule_xscale, self.ferrule_yscale]
        plate_scale_filename = os.path.join(self.ferrule_work_base, "ferrule_plate_scale.txt")
        np.savetxt(plate_scale_filename, np.array(self.ferrule_plate_scale))

        return

    def ferrule_create_gain(self):
        """Creates average gain table.
        Also attempts to determine position of ferrule in image.
        """
        # Subtract dark and normalize
        dsub = self.avg_flat - self.avg_dark
        dsub /= np.median(dsub)
        # Ferrule image is correctly-oriented. Translate camera to match
        plate_scale = self.ferrule_plate_scale
        if self.camera_rotation:
            dsub = np.rot90(dsub)
            plate_scale = plate_scale[::-1] # Reverse
        for translation in self.channel_translation:
            if translation.lower() == "rot90":
                dsub = np.rot90(dsub)
                plate_scale = plate_scale[::-1]
            elif translation.lower() == "flip":
                dsub = np.flip(dsub)
                plate_scale = plate_scale[::-1]
            elif translation.lower() == "fliplr":
                dsub = np.fliplr(dsub)
            elif translation.lower() == "flipud":
                dsub = np.flipud(dsub)
        # Fetch reference image of fiber bundle
        refim, refhdr = self.read_ferrule_map()
        # Interpolate reference image to observation plate scale
        # (if the observing setup is anything other than default)
        if self.francis_setup.lower() != "default":
            # Interpolate here... We *should* use the scipy interpolation functions.
            # But I'm writing this while exhausted, so we're using the sunpy reproject_to
            refmap = smap.Map(refim, refhdr)
            dat_dict = {
                "CDELT1": self.ferrule_xscale,
                "CDELT2": self.ferrule_yscale,
                "CTYPE1": "HPLN-TAN",
                "CTYPE2": "HPLT-TAN",
                "CUNIT1": "arcsec",
                "CUNIT2": "arcsec",
                "CRVAL1": 0,
                "CRVAL2": 0,
                "CRPIX1": dsub.shape[1] // 2,
                "CRPIX2": dsub.shape[0] // 2,
                "CROTA2": 0
            }
            datmap = smap.Map(dsub, dat_dict)
            refim = refmap.reproject_to(datmap.wcs).data
        # Divide out the fiber head image and set the gain table.
        # Save the fiber bundle position for later.
        master_shifts = np.zeros(2)
        sh_im = dsub.copy()
        for i in range(3):
            sh_im, shifts = align._image_align(
                sh_im, refim
            )
            master_shifts += shifts
        shifted_ferrule = scind.shift(refim, -master_shifts, mode="nearest")
        dummy = np.ones(dsub.shape)
        dummy[:shifted_ferrule.shape[0], :shifted_ferrule.shape[1]] = shifted_ferrule
        self.ferrule_position = [dummy.shape[1] // 2 - master_shifts[0], dummy.shape[0]//2 - master_shifts[1]]
        self.gain = (dsub / dummy) / np.median(dsub / dummy)
        return

    def read_ferrule_map(self, path: str="", fname: str="") -> tuple[np.ndarray, fits.Header]:
        """Reads template ferrule map for fiber bundle position.
        The template map has the fiber head centered in the middle of a
        1k x 1k image with a resolution of ~0.093"/pixel

        Parameters
        ----------
        path : str, optional
            If other than the default map, directory with alternate, by default ""
        fname : str, optional
            If other than the default map, filename of alternate, by default ""

        Returns
        -------
        fiber_head: np.ndarray
            Image of fiber head
        fiber_info: astropy.io.fits.Header
            Header containing spacing information
        """
        def read_data(datapath: str, filename: str) -> fits.hdu.hdulist.HDUList:
            with resources.path(datapath, filename) as df:
                return fits.open(df)

        if path == fname == "":
            mapfile = read_data("ssosoft.ifu.fiber_maps", "ferrule_map_20250923.fits")
            self.logger.info(
                "Attempting to load default fiber head image."
            )
        else:
            mapfile = fits.open(os.path.join(path, fname))
            self.logger.info(
                f"Loading fiber map {fname} from {path}"
            )
        try:
            fiber_head = mapfile[0].data
            fiber_info = mapfile[0].header
            mapfile.close()
        except (FileNotFoundError, IndexError) as err:
            raise("Improperly formatted fiber map file or file not found", err)
        else:
            return fiber_head, fiber_info

    def ferrule_save_fits_image(self, image:np.ndarray, file:str, header:dict | None=None, clobber:bool=True) -> None:
        """Saves image to FITS file

        Parameters
        ----------
        image : np.ndarray
            Image data
        file : str
            Path to save file to
        header : dict | None, optional
            If given, writes all key/value pairs to header, by default None
        clobber : bool, optional
            Overwrite existing file if true, by default True
        """
        hdu = fits.PrimaryHDU(image)
        if header is not None:
            for key in header.keys():
                hdu.header[key] = header[key]
        hdul = fits.HDUList([hdu])
        try:
            hdul.writeto(file, overwrite=clobber)
        except Exception as err:
            self.logger.warning(f"Could not write FITS file: {file}")
            self.logger.warning(f"With exception {err}")
        return


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

        self.dcss_lat = np.array(lat)
        self.dcss_lon = np.array(lon)
        self.dcss_times = np.array(timestamp, dtype="datetime64[ms]")
        self.dcss_see = np.array(scin)
        self.dcss_llvl = np.array(llvl)
        self.dcss_sdim = np.array(srad) / 2
        self.dcss_gdran = np.array(gdran) - 13.3
        return

    def ferrule_save_processed_maps(self) -> None:
        """Main cal loop for processed maps"""
        def rosa_print_average_image_progress():
            if not nfile % 10:
                self.logger.info("Progress: "
                                 "{:0.1%}.".format(nfile / len(ferrule_data_list))
                                 )
        ferrule_data_list = sorted(glob.glob(
            os.path.join(self.ferrule_data_base, self.ferrule_data_file_pattern)
        ))
        file_counter = 0
        for nfile, file in enumerate(
            tqdm.tqdm(
                ferrule_data_list, desc="Running Ferrule Calibration Loop",
                disable=not self.progress
            )
        ):
            with fits.open(file) as hdul:
                exptime = 1000 * float(hdul[0].header["EXPOSURE"].split(" ")[0])
                exptime_td = np.timedelta64(int(exptime), "ms")
                for hdu in hdul[1:]:
                    data = self.translate_image(hdu.data - self.avg_dark) / self.gain
                    mfgs = _mfgs(data)
                    startobs = np.datetime64(hdu.header["DATE"])
                    if self.correct_time:
                        startobs = startobs - np.timedelta64(1, "h")
                    endobs = startobs + exptime_td
                    header = {
                        "STARTOBS": startobs.astype(str),
                        "DATE-OBS": startobs.astype(str),
                        "ENDOBS": endobs.astype(str),
                        "DATE-END": endobs.astype(str),
                        "MFGS" : (round(mfgs, 5), "Deng et. al., 2015")
                    }
                    if self.dcss_log != "":
                        dcss_idx = _find_nearest(
                            self.dcss_times,
                            startobs
                        )
                        slon = self.dcss_lon[dcss_idx]
                        slat = self.dcss_lat[dcss_idx]
                        rotan = self.dcss_gdran[dcss_idx]
                        solrad = self.dcss_sdim[dcss_idx]
                        llvl = self.dcss_llvl[dcss_idx]
                        scin = self.dcss_see[dcss_idx]
                        telescope_pointing = SkyCoord(
                            slon * u.degree,
                            slat * u.degree,
                            frame=frames.HeliographicStonyhurst,
                            observer="earth",
                            obstime=self.dcss_times[dcss_idx].astype(str)
                        ).transform_to(frames.Helioprojective)
                        header["CRVAL1"] = round(telescope_pointing.Tx.value, 3)
                        header["CRVAL2"] = round(telescope_pointing.Ty.value, 3)
                        header["CTYPE1"] = "HPLN-TAN"
                        header["CTYPE2"] = "HPLT-TAN"
                        header["CUNIT1"] = "arcsec"
                        header["CUNIT2"] = "arcsec"
                        header["CDELT1"] = round(self.ferrule_plate_scale[0], 3)
                        header["CDELT2"] = round(self.ferrule_plate_scale[1], 3)
                        header["CRPIX1"] = data.shape[1] / 2
                        header["CRPIX2"] = data.shape[0] / 2
                        header["CROTA2"] = round(rotan, 3)
                        header["SCINT"] = (scin, "DST Seykora Scint.")
                        header["LLVL"] = (llvl, "DST Light Level")
                        header["RSUN_ARC"] = (solrad, "Radius of sun [arcsec]")
                    header["FERR-X"] = (self.ferrule_position[0], "Pixel x-pos of ferrule")
                    header["FERR-Y"] = (self.ferrule_position[1], "Pixel y-pos of ferrule")
                    header["PRSTEP1"] = ("DARK-SUBTRACTION", "FerruleCal/SSOSoft")
                    header["PRSTEP2"] = ("FLATFIELDING", "FerruleCal/SSOSoft")
                    header["PRSTEP3"] = ("TRANSLATE", "Translate to telescope optical axis")
                    file_pattern = f"{self.ferrule_obsdate}_{self.ferrule_obstime}_FERRULE_{file_counter:05d}.fits"
                    filename = os.path.join(self.ferrule_calibrated_base, file_pattern)
                    self.ferrule_save_fits_image(data, filename, header=header)
                    file_counter += 1
                rosa_print_average_image_progress()
        return
