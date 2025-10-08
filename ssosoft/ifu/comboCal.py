import configparser
import glob
import logging
import os

import astropy.io.fits as fits
import astropy.units as u
import numpy as np
import sunpy.coordinates.frames as frames
import sunpy.map as smap
import tqdm
from astropy.coordinates import SkyCoord
from sunpy.coordinates import RotatedSunFrame

from ssosoft.tools import alignment_tools as align

from . import ferruleCal, francisCal


def _find_nearest(array, value):
    """Finds index of closest value in array"""
    return np.abs(array-value).argmin()


class CombineCalibration:
    """
    Class for combining reduced Fiber data and Ferrule data with a
    context ROSA/HARDcam channel to constitute a single data product.

    Methods of this class will align the ferrule data with ROSA,
    and update the WCS values of the fiber data as well.
    """
    from ssosoft import ssosoftConfig

    def __init__(
        self,
        franciscal:francisCal.FrancisCal,
        ferrulecal:ferruleCal.FerruleCal
    ):
        """Class initialization

        Parameters
        ----------
        FrancisCal : francisCal.FrancisCal
            Instance of FrancisCal class with all class variables set
            (as if the calibration had just been run)
        FerruleCal : ferruleCal.FerruleCal
            Instance of FerruleCal class with all class variables set
            (as if the calibration had just been run)
        """
        self.francis_calbase = franciscal.calibrated_base
        self.francis_filelist = sorted(glob.glob(
            os.path.join(self.francis_calbase, "*.fits")
        ))
        if len(self.francis_filelist) == 0:
            raise FileNotFoundError(
                f"No reduced FRANCIS data found in {self.francis_calbase}."
            )
        self.francis_workbase = franciscal.work_base
        self.obsdate = franciscal.date
        self.obstime = franciscal.time
        self.francis_spectral_window = franciscal.wavelength
        self.ferrule_calbase = ferrulecal.ferrule_calibrated_base
        self.ferrule_filelist = sorted(glob.glob(
            os.path.join(self.ferrule_calbase, "*.fits")
        ))
        if len(self.ferrule_filelist) == 0:
            raise FileNotFoundError(
                f"No reduced ferrule imager data found in {self.ferrule_calbase}"
            )
        self.ferrule_plate_scale = ferrulecal.ferrule_plate_scale
        self.config_file = franciscal.config_file
        self.channel_translations = ferrulecal.channel_translation

        self.ferrule_target = ferrulecal.ferrule_target_file
        self.ferrule_burst_type = ferrulecal.burst_type # "registration" or "continuous"

        self.context_channel_name = ""
        self.context_channel_location = ""
        self.context_channel_filelist = []
        self.context_channel_target = ""

        self.progress = franciscal.progress

        self.coalign = True
        self.cleanup = True # Removes ferrule images after combination, as they cease to be relevant

        self.final_file_pattern = ""
        self.final_directory = os.path.join(self.francis_workbase, "combined")
        if not os.path.isdir(self.final_directory):
            print(f"{__name__}: os.mkdir: attempting to create directory {self.final_directory}")
            try:
                os.mkdir(self.final_directory)
            except Exception as err:
                print(f"An exception was raised: {err}")
                raise

        self.ferrule_coord = 0
        self.francis_coord = 0

        self.theta = 0
        self.offset = (0, 0)

        return

    def combine_calibration(self) -> None:
        """Function block of cal loop"""
        self.configure_run()
        self.register_ferrule_cam()
        self.write_combined_data_product()
        if self.cleanup and self.ferrule_calbase != "":
            self.logger.info(f"Cleaning up {self.ferrule_calbase}...")
            for file in self.ferrule_filelist:
                os.remove(file)
            os.rmdir(self.ferrule_calbase)

        return

    def configure_run(self) -> None:
        """Finds the context channel images, target, and
        reads some crucial data into (such as starttimes and coords)
        """
        config = configparser.ConfigParser()
        config.read(self.config_file)

        self.logfile = os.path.join(self.francis_workbase, f"{self.obstime}_FRANCIS_COMBINE.log")
        logging.config.fileConfig(self.config_file, defaults={"logfilename": self.logfile})
        self.logger = logging.getLogger("FRANCIS_COMBINE_Log")
        # Intro message
        self.logger.info(f"This is SSOsoft version {self.ssosoftConfig.__version__}")
        self.logger.info(f"Contact {self.ssosoftConfig.__email__} to report bugs, make suggestions, "
                            "of contribute")
        self.logger.info("Now configuring this FRANCIS calibration registration and calibration.")

        self.cleanup = config["SHARED"]["cleanup"] if "cleanup" in config["SHARED"].keys() \
            else self.cleanup
        if str(self.cleanup).lower() == "false":
            self.cleanup = False
        else:
            self.cleanup = True

        self.final_file_pattern = config["SHARED"]["finalFilePattern"]

        self.context_channel_name = config["SHARED"]["ROSAReferenceChannel"]
        self.context_channel_location = config["SHARED"]["ROSAReferenceLocation"]
        destretch_flist = sorted(glob.glob(
            os.path.join(self.context_channel_location, "splineDestretch", "*.fits")
        ))
        postspeckle_flist = sorted(glob.glob(
            os.path.join(self.context_channel_location, "postSpeckle", "*.fits")
        ))
        if len(destretch_flist) == len(postspeckle_flist) == 0:
            raise FileNotFoundError(
                f"No reduced files found in {self.context_channel_location}!"
            )
        elif len(destretch_flist) > 0:
            self.context_channel_filelist = destretch_flist # Default to more highly-processed
        else:
            self.context_channel_filelist = postspeckle_flist

        context_channel_target = os.path.join(
            self.context_channel_location, f"{self.context_channel_name}_orientation_target.fits"
        )
        if not os.path.exists(context_channel_target):
            self.logger.critical(
                f"No {self.context_channel_name} target file found: {context_channel_target}"
            )
            self.logger.critical(
                "A correctly-orientated target image is required for channel co-alignment."
            )
            self.logger.critical(
                "We will proceed with calibration, but the coordinates will be incorrect."
            )
        else:
            self.context_channel_target = context_channel_target
        return

    def register_ferrule_cam(self) -> None:
        """
        Performs registration between ferrule camera and ROSA/HARDcam reference cam
        """

        if self.context_channel_target == "" or self.ferrule_target == "":
            self.logger.critical(
                "No target images found. Proceeding without rigid alignment"
            )
            return

        offsets, rotation = align.align_derotate_channel_images(
            self.context_channel_target,
            self.ferrule_target
        )

        self.theta += rotation
        self.logger.info(
            f"Relative rotation between FRANCIS ferrule and {self.context_channel_name}: {self.theta}"
        )
        # Since we're adjusting the CRPIX instead of actually applying a shift,
        # we need to flip the sign of the offsets.
        # e.g., if we need to move down and to the right, the reference pixel has
        # move up and to the left.
        self.offset = -offsets
        self.logger.info(
            f"Offset [pixels] between FRANCIS ferrule and {self.context_channel_name}: {self.offset}"
        )
        return

    def write_combined_data_product(self) -> None:
        """Reads and combined FRANCIS, Ferrule, and reference ROSA files,
        Applies coalignment, and writes all products to a series of combined files.
        """
        if self.ferrule_burst_type.lower() == "reference":
            # Case: Ferrule data were taken as a single burst up top.
            # Find the best image, and use as the single reference throughtout
            mfgs = []
            for file in self.ferrule_filelist:
                with fits.open(file) as hdul:
                    mfgs.append(hdul[0].header["MFGS"])
            max_idx = np.nanargmax(np.array(mfgs))
            self.logger.info(
                "Ferrule camera was run in reference mode. Choosing a single context image."
            )
            self.logger.info(
                f"The clearest image was: {self.ferrule_filelist[max_idx]}"
            )
            self.logger.info(
                f"With a MFGS of {mfgs[max_idx]}"
            )

            reference_ferrule_files = [self.ferrule_filelist[max_idx]] * len(self.francis_filelist)
        elif self.ferrule_burst_type.lower() == "continuous":
            # Case: Ferrule data were taken continuously during FRANCIS acquisition.
            # Find the best ferrule image taken between each FRANCIS starttime/endtime
            ferrule_startobs = []
            mfgs = []
            for file in self.ferrule_filelist:
                with fits.open(file) as hdul:
                    ferrule_startobs.append(hdul[0].header["DATE-OBS"])
                    ferrule_startobs.append(hdul[0].header["MFGS"])
            ferrule_startobs = np.array(ferrule_startobs, dtype="datetime64[ms]")
            mfgs = np.array(mfgs)
            reference_ferrule_files = []
            for file in self.francis_filelist:
                with fits.open(file) as hdul:
                    startobs = np.datetime64(hdul[0].header["STARTOBS"])
                    endobs = np.datetime64(hdul[0].header["ENDOBS"])
                    start_idx = _find_nearest(ferrule_startobs, startobs)
                    end_idx = _find_nearest(ferrule_startobs, endobs)
                    if start_idx == end_idx:
                        reference_ferrule_files.append(self.ferrule_filelist[start_idx])
                    else:
                        # Best MFGS between start_idx and end_idx
                        idx = start_idx + np.nanargmax(mfgs[start_idx:end_idx])
                        reference_ferrule_files.append(self.ferrule_filelist[idx])
        else:
            self.logger.warning(
                "The acquisitionType keyword in the configuration file is improperly set!"
            )
            self.logger.warning(
                f"Provided value: {self.ferrule_burst_type}. Allowed values are \"reference\" or \"continuous\""
            )
            self.logger.warning(
                "We will be proceeding using the last ferrule image taken."
            )
            reference_ferrule_files = [self.ferrule_filelist[-1]] * len(self.francis_filelist)
        # Assemble startobs of all reference channel data
        reference_channel_startobs = []
        for file in self.context_channel_filelist:
            with fits.open(file) as hdul:
                reference_channel_startobs.append(hdul[0].header["DATE-BEG"])
        reference_channel_startobs = np.array(reference_channel_startobs, dtype="datetime64[ms]")
        # Grab the reference coordinate from the 0th reference file
        # We'll use this as the master CRVAL coordinate for the ferrule cam
        # From that, we'll back out the coordinate of the fiber head itself.
        refmap = smap.Map(self.context_channel_filelist[0])
        # Since we did the channel coalignment in the ROSA reductions by updating CRPIX1/2
        # and CROTA2, we can't just grab the CRVAL1/2 keywords. These will not refer to the
        # map center, unless the channel chosen for inclusion is the same as the channel used
        # for the ROSA coalignment, and we used the map center for the coalignment with FRANCIS.
        # Instead, we'll pop the 0th file into a sunpy map and grab the center coordinate.
        # That will refer to the map center, not the CRVAL reference coordinate.
        ref_coord = refmap.center
        with fits.open(reference_ferrule_files[0]) as hdul:
            dummy_hdr = {
                "CRPIX1": hdul[0].header["CRPIX1"] + self.offset[1],
                "CRPIX2": hdul[0].header["CRPIX2"] + self.offset[0],
                "CROTA2": hdul[0].header["CROTA2"] + self.theta,
                "CTYPE1": "HPLN-TAN",
                "CTYPE2": "HPLT-TAN",
                "CUNIT1": "arcsec",
                "CUNIT2": "arcsec",
                "CDELT1": hdul[0].header["CDELT1"],
                "CDELT2": hdul[0].header["CDELT2"],
                "CRVAL1": ref_coord.Tx.value,
                "CRVAL2": ref_coord.Ty.value,
                "DATE-OBS": ref_coord.obstime.value
            }
            ferrule_map = smap.Map(hdul[0].data, dummy_hdr)
            fiber_coord = ferrule_map.wcs.pixel_to_world(
                hdul[0].header["FERR-X"] * u.pix, hdul[0].header["FERR-Y"] * u.pix
            )

        self.logger.info("Beginning channel combine and registration.")
        def print_progress():
            if not fnum % 10:
                self.logger.info(
                    f"Progress: {fnum / len(self.francis_filelist):0.1%}"
                )

        # Main loop for combine and pointing update:
        for fnum, file in enumerate(tqdm.tqdm(
            self.francis_filelist,
            desc=f"Combining FRANCIS and {self.context_channel_name} date.",
            disable=not self.progress
        )):
            # Grab all the data we need
            with fits.open(file) as hdul:
                francis_data = hdul[1].data.copy()
                francis_hdr = hdul[1].header.copy()
                wvl_data = hdul[2].data.copy()
                wvl_hdr = hdul[2].header.copy()
                master_hdr = hdul[0].header.copy()
                reverse = 0
                for translation in self.channel_translations:
                    if translation.lower() == "rot90":
                        francis_data = np.rot90(francis_data, axes=(0, 1))
                        reverse += 1
                    elif translation.lower() == "flip":
                        francis_data = np.flip(francis_data, axis=(0, 1))
                    elif translation.lower() == "fliplr":
                        francis_data = np.flip(francis_data, axis=1)
                    elif translation.lower() == "flipud":
                        francis_data = np.flip(francis_data, axis=0)

            with fits.open(reference_ferrule_files[fnum]) as hdul:
                ferrule_data = hdul[0].data.copy()
                ferrule_hdr = hdul[0].header.copy()
            # near-cotemporal reference channel data:
            tidx = _find_nearest(reference_channel_startobs, np.datetime64(master_hdr["DATE-OBS"]))
            with fits.open(self.context_channel_filelist[tidx]) as hdul:
                context_data = hdul[0].data.copy()
                context_hdr = hdul[0].header.copy()
            # Differentially rotate reference points:
            dt = (
                np.datetime64(master_hdr["DATE-OBS"]) - np.datetime64(ref_coord.obstime.value)
            ).astype("timedelta64[ms]").astype(int) * u.ms
            rotated_refpoint = SkyCoord(
                RotatedSunFrame(
                    base=ref_coord, duration=dt
                ), observer="earth", obstime=master_hdr["DATE-OBS"]
            ).transform_to(frames.Helioprojective)
            rotated_fpoint = SkyCoord(
                RotatedSunFrame(
                    base=fiber_coord, duration=dt
                ), observer="earth", obstime=master_hdr["DATE-OBS"]
            )
            prsteps = len([i for i in master_hdr.keys() if "PRSTEP" in i])
            master_hdr[f"PRSTEP{prsteps+1}"] = (
                "IMCOMBINE", f"Register to Ferrule camera and {self.context_channel_name}"
            )
            master_hdr[f"PRSTEP{prsteps+1}"] = (
                "SOLAR-ALIGN", f"Align to {self.context_channel_name} via Ferrule camera"
            )
            master_hdr["COMMENT"] = f"Pointing updated on {np.datetime64("now")}"

            master_hdr["XCEN"] = (round(rotated_fpoint.Tx.value, 3), "Fiber head center")
            master_hdr["YCEN"] = (round(rotated_fpoint.Ty.value, 3), "Fiber head center")
            master_hdr["ROT"] += round(self.theta, 3)
            # If we rotated the image, we need to reverse the X/Y scales
            if reverse % 2 != 0:
                fovx = master_hdr["FOVX"]
                fovy = master_hdr["FOVY"]
                master_hdr["FOVY"] = fovx
                master_hdr["FOVX"] = fovy
                master_hdr["COMMENT"] = "Fiber head is rotated! CDELT and FOV keywords have been changed but not spacing keywords!"

            ext0 = fits.PrimaryHDU()
            ext0.header = master_hdr

            ext_fibers = fits.ImageHDU(francis_data)
            ext_fibers.header = francis_hdr
            ext_fibers.header["CRVAL1"] = (round(rotated_fpoint.Tx.value, 3), "Fiber head center")
            ext_fibers.header["CRVAL2"] = (round(rotated_fpoint.Ty.value, 3), "Fiber head center")
            ext_fibers.header["CROTA2"] += round(self.theta, 3)
            if reverse % 2 != 0:
                dx = ext_fibers["CDELT1"]
                dy = ext_fibers["CDELT2"]
                ext_fibers["CDELT2"] = dx
                ext_fibers["CDELT1"] = dy
                ext_fibers["COMMENT"] = "Fiber bundle rotated! Be wary of alignment!"

            ext_wvl = fits.ImageHDU(wvl_data)
            ext_wvl.header = wvl_hdr

            ext_ferrule = fits.ImageHDU(ferrule_data)
            for key in ferrule_hdr.keys():
                ext_ferrule.header[key] = (ferrule_hdr[key], ferrule_hdr.comments[key])
            # ext_ferrule.header = ferrule_hdr
            ext_ferrule.header["EXTNAME"] = "FERRULE"
            ext_ferrule.header["CROTA2"] += self.theta
            ext_ferrule.header["CRVAL1"] = round(rotated_refpoint.Tx.value, 3)
            ext_ferrule.header["CRVAL2"] = round(rotated_refpoint.Ty.value, 3)
            ext_ferrule.header["CRPIX1"] += self.offset[1]
            ext_ferrule.header["CRPIX2"] += self.offset[0]
            ext_ferrule.header["DATE-ACT"] = (
                ext_ferrule.header["STARTOBS"], "Actual startobs; DATE-OBS keywords altered for pointing."
            )
            ext_ferrule.header["DATE-OBS"] = rotated_refpoint.obstime.value
            ext_ferrule.header["DATE-END"] = rotated_refpoint.obstime.value
            ext_ferrule.header["STARTOBS"] = rotated_refpoint.obstime.value
            ext_ferrule.header["ENDOBS"] = rotated_refpoint.obstime.value

            prsteps = len([i for i in ext_ferrule.header.keys() if "PRSTEP" in i])
            ext_ferrule.header[f"PRSTEP{prsteps + 1}"] = ("CHANNEL-REG", f"Register to {self.context_channel_name}")

            ext_ctx = fits.ImageHDU(context_data)
            for key in context_hdr.keys():
                if "COMMENT" not in key:
                    ext_ctx.header[key] = (context_hdr[key], context_hdr.comments[key])
                if any(["COMMENT" in key for key in context_hdr]):
                    comments = str(context_hdr["COMMENT"]).split("\n")
                    for comment in comments:
                        ext_ctx.header["COMMENT"] = comment
            # ext_ctx.header = context_hdr
            ext_ctx.header["EXTNAME"] = "REFERENCE"

            fits_hdul = fits.HDUList(
                [ext0, ext_fibers, ext_wvl, ext_ferrule, ext_ctx]
            )
            outname = self.final_file_pattern.format(
                self.obsdate, self.obstime,
                self.francis_spectral_window,
                self.context_channel_name,
                fnum
            )
            outfile = os.path.join(self.final_directory, outname)
            fits_hdul.writeto(outfile, overwrite=True)
            print_progress()
        return
