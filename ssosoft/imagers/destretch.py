import configparser
import glob
import os
import warnings

import astropy.io.fits as fits
import numpy as np
import scipy.ndimage as scind
import tqdm

from ..tools import alignment_tools as align
from ..tools import movie_makers as movie
from .pyDestretch import Destretch


class RosaZylaDestretch:
    """
    Master destretch class for SSOC imagers.

    I've decided to remove the functionality for destretching
    one imager relative to a difference channel.

    The reasons for this change include:
        1.) While the destretch is, indeed, slow, it's mostly the Zyla destretch
            that's slow. ROSA destretch is on the order of a couple hours
        2.) A major component of the jitter includes shifting in the subfields
            during the speckling process. This is different between different
            channels, and can result in some odd jittering during reconstruction.

    Instead, I've implemented a better determination of plate scale during the
    initial calibration stage, as well as the ability to write AF resolution target
    images to the reduced directory. From this, we'll use bits of the image alignment
    routines we have in stock to figure out the relative shift and rotation between
    channels.

    Rather than interpolating or shifting the images further, we'll apply the shifts
    to the CRPIX1/2 WCS keywords, and the rotations to the CROTA2 WCS keyword.

    That way, when the reduced files are parsed using Sunpy or another WCS handler,
    they'll be aligned without the need to alter the image data too much.

    The exception is the bulk translations and 90-degree rotations, which preserve
    the image integrity.

    Unlike kisipWrapper, we aren't going to take the set up rosaZylaCal class as an
    input. That would make it much easier, however, the class has to stand alone due
    to the historical bulk of
    """
    def __init__(self, instruments: str | list, config_file: list) -> None:
        """Class init.

        Parameters
        ----------
        instruments : str | list
            Instrument or list of instruments to repack and destretch
        config_file : list
            Path to config file
        """

        self.config_file = config_file
        self.instruments = instruments
        if type(self.instruments) is str:
            self.instruments = [self.instruments]
        self.reference_channel = ""
        self.work_base = ""
        self.dstr_from = ""
        self.hdr_base = ""
        self.speckle_base = ""
        self.speckle_file_form = ""
        self.postspeckle_base = ""
        self.postdeflow_base = ""
        self.postdestretch_base = ""
        self.kernels = [0]
        self.dstr_base = ""
        self.dstr_flows = ""
        self.theta = 0
        self.solar_rotation_error = 0
        self.wave = 0
        self.offset = (0, 0)
        self.dstr_method = ""
        self.dstr_window = 0
        self.channel = ""
        self.datashape = (0, 0)
        self.flow_window = 0
        self.dstr_file_pattern = ""
        self.repair_tolerance = 0.5

        self.reference_channel_working_directory = ""

        self.master_alignment = "HMI"
        self.verify_alignment = True

        self.date = ""
        self.time = ""

        self.dstr_filelist = []
        self.pspkl_filelist = []
        self.deflow_filelist = []

        # We're no longer treating each ROSA channel as referenced to a single
        # channel, with ZYLA as a separate. Instead, each camera will have a
        # channelTranslation attribute to get it separately to the correct
        # orientation. I expect this to actually be easier.
        self.channel_translation = []

        self.scale_reverse = 0 # New param. With rotation and flips, the scale will swap. Tracks number of swaps

        self.progress = False
        self.solar_align = False
        self.reference_coordinate = {}

        self.create_context_movies = False
        self.context_movie_directory = ""

        return

    def assert_flist(self, flist):
        assert (len(flist) != 0), "List contains no matches"
        return

    def perform_destretch(self) -> None:
        """Performs SSOC destretch"""
        for channel in self.instruments:
            self.channel = channel
            self.configure_destretch()
            # Hardcoded data shape for now...
            if "ZYLA" in channel:
                self.datashape = (2048, 2048)
            else:
                self.datashape = (1002, 1004)
            if self.dstr_method.lower() == "speckle":
                self.speckle_to_fits()
            else:
                self.pspkl_filelist = sorted(glob.glob(
                    os.path.join(self.postspeckle_base, "*.fits")
                ))
            self.destretch_observation()
            if self.flow_window > 0:
                self.remove_flows()
                self.apply_flow_vectors()
        return

    def register_rosa_zyla_observations(self):
        """
        Run after perform_destretch to coalign and derotate all channels
        Performs optional solar alignment as well
        """
        for channel in self.instruments:
            self.channel = channel
            self.configure_destretch()
            self.apply_bulk_translations()
            if self.reference_channel != "" and self.channel == self.reference_channel and self.solar_align:
                self.update_reference_frame()
                self.apply_solar_offsets()
            if self.reference_channel != "" and self.channel != self.reference_channel:
                self.channel_offset_rotation()
                self.channel_apply_offsets()
                if self.solar_align:
                    self.apply_solar_offsets()
            if self.create_context_movies:
                self.generate_context_movie()
        return


    def configure_destretch(self) -> None:
        """
        Parses config file and sets up class instance with required variables
        """
        # Step 1: Save the channel within this function, and re-initialize class
        # Since the perform_destretch() function runs a loop with multiple calls
        # to this function, it's best to clear out all class variables with by
        # calling __init__ again to avoid any lingering mis-set values.
        # We do want to keep some values, however, such as the working channel,
        # and the updated solar reference pointing, if we've got it.
        channel = self.channel
        ref_coord = self.reference_coordinate = {}
        self.__init__(self.instruments, self.config_file)
        self.reference_coordinate = ref_coord
        self.channel = channel

        config = configparser.ConfigParser()
        config.read(self.config_file)

        # Directories
        self.work_base = config[self.channel]["workBase"]
        if not os.path.exists(self.work_base):
            raise FileNotFoundError(
                f"workBase: {self.work_base}, directory not found!"
                "Please run rosaZylaCal and kisipWrapper before destretch!"
            )
        self.hdr_base = os.path.join(self.work_base, "hdrs")
        self.speckle_base = os.path.join(self.work_base, "speckle")
        if not os.path.exists(self.speckle_base):
            raise FileNotFoundError(
                f"No directory containin speckle bursts found at: {self.speckle_base}"
            )
        self.speckle_file_form = config[self.channel]["speckledFileForm"]
        self.postspeckle_base = os.path.join(self.work_base, "postSpeckle")
        if not os.path.exists(self.postspeckle_base):
            os.mkdir(self.postspeckle_base)
        self.postdestretch_base = os.path.join(self.work_base, "splineDestretch")
        if not os.path.exists(self.postdestretch_base):
            print(
                f"{__name__}: os.mkdir: attempting to create directory: {self.postdestretch_base}"
            )
            os.mkdir(self.postdestretch_base)
        self.dstr_base = os.path.join(self.work_base, "destretch_vectors")
        if not os.path.exists(self.dstr_base):
            print(
                f"{__name__}: os.mkdir: attempting to create directory: {self.dstr_base}"
            )
            os.mkdir(self.dstr_base)

        # Destretch params
        self.dstr_from = config[self.channel]["dstrFrom"]
        self.dstr_method = config[self.channel]["dstrMethod"]
        self.dstr_window = int(config[self.channel]["dstrWindow"])
        self.kernels = [int(i) for i in config[self.channel]["dstrKernel"].split(",")]
        self.repair_tolerance = float(config[self.channel]["repairTolerance"]) if "repairtolerance" \
            in config[self.channel].keys() else self.repair_tolerance
        self.flow_window = int(config[self.channel]["flowWindow"])

        self.wave = float(config[self.channel]["wavelengthnm"])
        self.date = config[self.channel]["obsDate"]
        self.time = config[self.channel]["obsTime"]

        if self.flow_window > 0:
            self.postdeflow_base = os.path.join(self.work_base, "flowPreservedDestretch")
            if not os.path.exists(self.postdeflow_base):
                print(
                    f"{__name__}: os.mkdir: attempting to create directory: {self.postdeflow_base}"
                )
                os.mkdir(self.postdeflow_base)
            self.dstr_flows = os.path.join(self.work_base, "destretch_vectors_noflow")
            if not os.path.exists(self.dstr_flows):
                print(
                    f"{__name__}: os.mkdir: attempting to create directory: {self.dstr_flows}"
                )
        self.dstr_file_pattern = config[self.channel]["destretchedFileForm"]

        # Translation and alignment
        if "chaneltranslation" in config[self.channel].keys():
            trans_string = config[self.channel]["channelTranslation"]
            if trans_string != "" and trans_string.lower() != "none":
                self.channel_translation = trans_string.split(",")
        elif "bulktranslation" in config[self.channel].keys():
            warnings.warn(
                "Keyword \"bulkTranslation\" has been replaced by \"channelTranslation\"."
                "Please update your configuration file.",
                DeprecationWarning,
                stacklevel=2
            )
        if "SHARED" in config[self.channel].keys():
            self.reference_channel = config["SHARED"]["referenceChannel"] if "referencechannel" \
                in config["SHARED"].keys() else None
            if "referencechannel" in config["SHARED"].keys():
                self.reference_channel_working_directory = config[
                    config["SHARED"]["referenceChannel"]
                ]["workBase"]
            self.solar_align = config["SHARED"]["solarAlign"] if "solaralign" \
                in config["SHARED"].keys() else False
            self.solar_align = True if str(self.solar_align).lower() == "true" else False
            self.verify_alignment = config["SHARED"]["verfiypointingupdate"] if "verifypointingupdate" \
                in config["SHARED"].keys() else False
            self.verify_alignment = True if str(self.verify_alignment) == "true" else False
            self.create_context_movies = config["SHARED"]["contextMovies"] if "contextmovies" \
                in config["SHARED"].keys() else False
            self.create_context_movies = True if str(self.create_context_movies).lower() == "true" else False
            self.context_movie_directory = config["SHARED"]["contextMovieDirectory"] if \
                "contextmoviedirectory" in config["SHARED"].keys() else ""

        # Progress bars
        self.progress = config[self.channel]["progress"] if "progress" in config[self.channel].keys() \
            else False
        self.progress = True if str(self.progress).lower() == "true" else False

        return

    def speckle_to_fits(self) -> None:
        """Function to speckle/*.final files to FITS format in the postSpeckle directory"""

        spkl_flist = sorted(glob.glob(os.path.join(self.speckle_base, "*.final")))
        try:
            self.assert_flist(spkl_flist)
        except AssertionError as err:
            print(f"Error: speckle files: {err}")
            raise

        alpha_flist = sorted(glob.glob(os.path.join(self.speckle_base, "*.subalpha")))
        try:
            self.assert_flist(alpha_flist)
        except AssertionError as err:
            print(f"Error: subalpha files: {err}")
            raise

        hdr_list = sorted(glob.glob(os.path.join(self.hdr_base, "*.txt")))
        try:
            self.assert_flist(hdr_list)
        except AssertionError as err:
            print(f"Error: header files: {err}")
            raise

        if not len(spkl_flist) == len(alpha_flist) == len(hdr_list):
            raise ValueError(
                f"Unequal numbers of speckle ({len(spkl_flist)}) files, "
                f"subalpha ({len(alpha_flist)}), "
                f"and header ({len(hdr_list)}) files. "
                f"Check configuration of {self.work_base}."
            )


        for i, file in enumerate(
            tqdm.tqdm(spkl_flist, desc="Converting speckle to FITS", disable=not self.progress)
        ):
            speckle_image = np.fromfile(file, dtype=np.float32).reshape(self.datashape)
            avg_alpha = np.fromfile(alpha_flist[i], dtype=np.float32)[0]
            fname = os.path.join(
                self.postspeckle_base,
                os.path.split(file)[-1] + ".fits"
            )
            self.pspkl_filelist.append(self.write_fits(fname, speckle_image, hdr_list[i], alpha=avg_alpha, prstep=3))
        return

    def destretch_observation(self) -> None:
        """Performs destretch with user-configured parameters."""
        try:
            self.assert_flist(self.pspkl_filelist)
        except AssertionError as err:
            print(f"Error: postspeckle list: {err}")
            raise

        if self.dstr_method.lower() == "reference":
            with fits.open(self.pspkl_filelist[self.dstr_window]) as hdul:
                reference = fits.open(hdul[0]).data
            reference_cube = reference[np.newaxis, :, :]
        else:
            reference_cube = np.zeros((self.dstr_window, *self.datashape))
            reference = np.zeros(self.datashape)

        for i, file in enumerate(
            tqdm.tqdm(self.pspkl_filelist, desc=f"Destretching {self.channel}", disable=not self.progress)
        ):
            with fits.open(file) as hdul:
                img = hdul[0].data.copy()
                hdr = hdul[0].header.copy()
            if self.dstr_method.lower() == "running" and i == 0:
                reference_cube[0, :, :] = img
                reference = reference_cube[0]
            elif self.dstr_method.lower() == "running" and i == 1:
                reference = reference_cube[0, :, :]
            elif self.dstr_method.lower() == "running" and i < self.dstr_window:
                reference = np.nanmean(reference_cube[:i, :, :], axis=0)
            elif self.dstr_method.lower() == "running" and i > self.dstr_window:
                reference = np.nanmean(reference_cube, axis=0)

            dstr_object = Destretch(
                img,
                reference,
                self.kernels,
                return_vectors=True,
                repair_tolerance=self.repair_tolerance
            )
            dstr_im, dstr_vecs = dstr_object.perform_destretch()
            if type(dstr_vecs) is not list:
                dstr_vecs = [dstr_vecs]

            if self.dstr_method == "running":
                reference_cube[int(i % self.dstr_window), :, :] = dstr_im

            # Write FITS files and vectors
            dvec_name = os.path.join(self.dstr_base, str(i).zfill(5) + ".npz")
            np.savez(dvec_name, *dstr_vecs)

            fname = os.path.join(
                self.postdestretch_base,
                self.dstr_file_pattern.format(
                    self.date, self.time, i
                )
            )
            self.dstr_filelist.append(self.write_fits(fname, dstr_im, hdr, prstep=4))

        return

    def remove_flows(self) -> None:
        """Function to remove lateral solar flows from destretch vectors.
        The basic method is to load the target control points from the
        saved destretch parameters, then doing a median filter on the cumulative
        sum of the shifts in the time direction.
        We then take a mean filter (uniform filter) of the median filter.
        These are our flows.

        These flows subtracted off the cumulative sum of the shifts.
        The remainder becomes the final shifts.

        This currently only runs over the last kernel in the sequence.

        At the moment, I am not particularly happy with the results this
        method produces. If any of my successors have a bright and shining
        idea on the best method for this, please let me know.
        """
        vector_flist = sorted(glob.glob(os.path.join(self.dstr_base, "*.npz")))
        try:
            self.assert_flist(vector_flist)
        except AssertionError as err:
            print(f"Error: destretch vector files: {err}")
            raise

        template_file = np.load(vector_flist[0])
        template_vecs = [template_file[k] for k in template_file] # unpack from npz
        fine_scale_shape = template_vecs[-1].shape
        shifts_all = np.zeros((len(vector_flist), *fine_scale_shape))
        for i, file in enumerate(tqdm.tqdm(vector_flist, desc="Loading destretch vectors", disable=not self.progress)):
            dstr = np.load(file)
            vectors = [dstr[k] for k in dstr]
            shifts_all[i] = vectors[-1]
        # ...*cumulative* sum
        shifts_cumsum = np.cumsum(shifts_all, axis=0)
        median_filtered = scind.median_filter(
            shifts_cumsum,
            size=(self.flow_window, 1, 1, 1),
            mode="nearest"
        )
        uniform_filtered = scind.uniform_filter(
            median_filtered,
            size=(self.flow_window, 1, 1, 1),
            mode="nearest"
        )
        flow_detr_shifts = shifts_cumsum - uniform_filtered

        for i, file in enumerate(
            tqdm.tqdm(vector_flist, desc="Saving flow-detrended vectors", disable=not self.progress)
        ):
            original_file = np.load(file)
            original_arrays = [original_file[k] for k in original_file]
            original_arrays[-1] = flow_detr_shifts[i]
            write_file = os.path.join(self.dstr_flows, str(i).zfill(5))
            np.savez(write_file + ".npz", *original_arrays)
        return

    def apply_flow_vectors(self) -> None:
        """Apply flow-removed destretch vectors, write the fits files to new directory."""
        if self.pspkl_filelist == []:
            self.pspkl_filelist = sorted(glob.glob(os.path.join(self.postspeckle_base, "*.fits")))
        try:
            self.assert_flist(self.pspkl_filelist)
        except AssertionError as err:
            print(f"Error: postspeckle list: {err}")
            raise

        deflow_flist = sorted(glob.glob(os.path.join(self.dstr_flows, "*.npz")))
        try:
            self.assert_flist(deflow_flist)
        except AssertionError as err:
            print(f"Error: deflow vector list: {err}")

        for i, file in enumerate(
            tqdm.tqdm(self.pspkl_filelist, desc="Applying de-flowed destretch", disable=not self.progress)
        ):
            with fits.open(file) as hdul:
                im = hdul[0].data.copy()
                hdr = hdul[0].header.copy()
            dstr_vecs = np.load(deflow_flist[i])
            vecs = [dstr_vecs[k] for k in dstr_vecs]
            d = Destretch(im, im, self.kernels, warp_vectors=vecs)
            dstrim = d.perform_destretch()
            fname = os.path.join(
                self.postdeflow_base,
                self.dstr_file_pattern.format(self.date, self.time, i)
            )
            self.deflow_filelist.append(self.write_fits(fname, dstrim, hdr, prstep=5))
        return

    def write_fits(
            self,
            fname: str, data: np.ndarray, hdr: str | fits.header.Header,
            alpha: float | None=None, prstep:int=4, overwrite:bool=True
    ) -> str:
        """Writes FITS file with given data and auxillary header information

        Parameters
        ----------
        fname : str
            Filename to write file to.
        data : np.ndarray
            2D numpy array with data to save
        hdr : str | fits
            Either a text file containing header information or a formatted fits header
        alpha : float | None, optional
            If given, the average alpha for the frame from speckle
        prstep : int, optional
            Index of the prstep keyword to use, by default 4

        Returns
        -------
        str
            Filename of saved FITS file
        """

        allowed_keywords = [
            "DATE", "STARTOBS", "ENDOBS", "DATE-AVG",
            "EXPOSURE", "XPOSUR", "TEXPOSUR", "NSUMEXP"
            "HIERARCH",
            "CRVAL1", "CRVAL2",
            "CTYPE1", "CTYPE2",
            "CUNIT1", "CUNIT2",
            "CDELT1", "CDELT2",
            "CRPIX1", "CRPIX2",
            "CROTA2",
            "SCINT", "LLVL",
            "RSUN_REF"
        ]
        float_keywords = [
            "CRVAL1", "CRVAL2",
            "CROTA2",
            "SCINT", "LLVL",
        ]
        asec_comment_keywords = [
            "CDELT1", "CDELT2",
            "CRPIX1", "CRPIX2",
            "RSUN_REF"
        ]
        prstep_flags = ["PRSTEP1", "PRSTEP2", "PRSTEP3", "PRSTEP4", "PRSTEP5"]
        prstep_values = [
            "DARK-SUBTRACTION,FLATFIELDING",
            "SPECKLE-DECONVOLUTION",
            "ALIGN TO SOLAR NORTH",
            "DESTRETCHING",
            "FLOW-PRESERVING-DESTRETCHING"
        ]
        prstep_comments = [
            "SSOsoft",
            "KISIP v0.6",
            "SSOsoft",
            "pyDestretch",
            "pyDestretch with flow preservation"
        ]
        if type(hdr) is fits.header.Header:
            hdul = fits.HDUList(fits.PrimaryHDU(data, header=hdr))
        else:
            hdul = fits.HDUList(fits.PrimaryHDU(data))
        hdul[0].header["BUNIT"] = "DN"
        hdul[0].header["AUTHOR"] = "sellers"
        hdul[0].header["TELESCOP"] = "DST"
        hdul[0].header["ORIGIN"] = "SSOC"
        if "ROSA" in self.channel.upper():
            hdul[0].header["INSTRUME"] = "ROSA"
        if "ZYLA" in self.channel.upper():
            hdul[0].header["INSTRUME"] = "HARDCAM"
        hdul[0].header["WAVE"] = self.wave
        hdul[0].header["WAVEUNIT"] = "nm"
        if type(hdr) is str:
            with open(hdr, "r") as file:
                lines = file.readlines()
                for line in lines:
                    slug = line.split("=")[0].strip()
                    field = line.split("=")[-1].split("/")[0]
                    field = field.replace("\n", "").replace("\'", "").strip()
                    if field.isnumeric():
                        field = float(field)
                    if any(substring in slug for substring in allowed_keywords):
                        if "STARTOBS" in slug:
                            hdul[0].header["DATE"] = (np.datetime64("now").astype(str), "Date of file creation")
                            hdul[0].header["STARTOBS"] = (field, "Date of start of observation")
                            hdul[0].header["DATE-BEG"] = (field, "Date of start of observation")
                        elif "ENDOBS" in slug:
                            hdul[0].header["ENDOBS"] = (field, "Date of end of observation")
                            hdul[0].header["DATE-END"] = (field, "Date of end of observation")
                        elif "DATE-AVG" in slug:
                            hdul[0].header["DATE-AVG"] = (field, "Average obstime")
                        elif "RSUN" in slug:
                            rsun = float(field)
                            if rsun > 700:
                                rsun /= 2
                            hdul[0].header["RSUN_ARC"] = (round(rsun, 3), "Radius of Sun in arcsec")
                        elif any(substring in slug for substring in float_keywords):
                            hdul[0].header[slug] = round(float(field), 3)
                        elif "EXPTIME" in slug:
                            hdul[0].header["EXPTIME"] = (round(float(field), 1), "ms, Single-frame exposure time")
                        elif "NSUMEXP" in slug:
                            hdul[0].header["NSUMEXP"] = (int(field), "Number of frames used in reconstruction")
                        elif "TEXPOSUR" in slug:
                            hdul[0].header["TEXPOSUR"] = (round(float(field), 1), "Total accumulation time")
                        elif any(substring in slug for substring in asec_comment_keywords):
                            hdul[0].header[slug] = (round(float(field), 3), "arcsec")
                        else:
                            hdul[0].header[slug] = field
        if alpha:
            hdul[0].header["SPKLALPH"] = alpha
        for i in range(prstep):
            hdul[0].header[prstep_flags[i]] = (prstep_values[i], prstep_comments[i])
        hdul.writeto(fname, overwrite=overwrite)
        return fname

    def apply_bulk_translations(self) -> None:
        """Performs per-camera sequence of translations specified in config file"""
        # First, search for post-speckle, spline destretched, and flow-preserve destretched files
        self.pspkl_filelist = sorted(glob.glob(os.path.join(self.postspeckle_base, "*.fits")))
        self.dstr_filelist = sorted(glob.glob(os.path.join(self.postdestretch_base, "*.fits")))
        if self.flow_window > 0:
            self.deflow_filelist = sorted(glob.glob(os.path.join(self.postdeflow_base, "*.fits")))

        if len(self.pspkl_filelist) == len(self.dstr_filelist) == len(self.deflow_filelist) == 0:
            raise FileNotFoundError("No processed files found!")

        full_filelist = self.pspkl_filelist + self.dstr_filelist + self.deflow_filelist
        for translation in self.channel_translation:
            if translation.lower() == "rot90" or translation.lower() == "flip":
                self.scale_reverse += 1
        master_header = ""
        for file in tqdm.tqdm(
            full_filelist,
            desc=f"Applying translations to {len(full_filelist)} reduced files",
            disable=not self.progress
        ):
            with fits.open(file, mode="update") as hdul:
                data = hdul[0].data.copy()
                xscale = hdul[0].header["CDELT1"]
                yscale = hdul[0].header["CDELT2"]
                scales = np.array([xscale, yscale]).copy()
                for translation in self.channel_translation:
                    if translation.lower() == "rot90":
                        data = np.rot90(data)
                    elif translation.lower() == "flip":
                        data = np.flip(data)
                    elif translation.lower() == "fliplr":
                        data = np.fliplr(data)
                    elif translation.lower() == "flipud":
                        data = np.flipud(data)
                # If the net total translation flipped the X/Y axes, reverse
                # the scales, so when we re-fill the header, the scales are correct.
                if self.scale_reverse % 2 == 1:
                    scales = scales[::-1]
                hdul[0].data = data
                hdul[0].header["CDELT1"] = scales[0]
                hdul[0].header["CDELT2"] = scales[1]
                if file == full_filelist[0]:
                    master_header = hdul[0].header.copy()
                    master_header["CRVAL1"] = 0.0
                    master_header["CRVAL2"] = 0.0
                    master_header["CROTA2"] = 0.0
                hdul.flush()
        # Translate and save the target image, if it exists with a sunpy-compatible header:
        target_file = os.path.join(self.work_base, f"{self.channel}_target.fits")
        if os.path.exists(target_file):
            with fits.open(target_file) as hdul:
                targ_im = hdul[0].data.copy()
            for translation in self.channel_translation:
                if translation.lower() == "rot90":
                    targ_im = np.rot90(targ_im)
                elif translation.lower() == "flip":
                    targ_im = np.flip(targ_im)
                elif translation.lower() == "fliplr":
                    targ_im = np.fliplr(targ_im)
                elif translation.lower() == "flipud":
                    targ_im = np.flipud(targ_im)
            fname = os.path.join(self.work_base, f"{self.channel}_orientation_target.fits")
            self.write_fits(fname, targ_im, master_header, prstep=0)
        else:
            warnings.warn(
                f"No target image found at {target_file}! Data cannot be registered to a reference channel!"
            )
        return

    def channel_offset_rotation(self) -> None:
        """
        Determines channel offset and relative rotation to the reference channel.
        We'll be using sunpy Maps. This allows us to crib its reproject_to function,
        which operates as shorthand for interpolation and cropping.

        Basically, we get two same-sized images of the same region to perform the alignment.

        This DOES rely on having a correctly-oriented target image in the working directory
        for each the reference and target channel. We'll throw an error if we don't have that.
        """

        reference_channel_target_file = os.path.join(
            self.reference_channel_working_directory,
            f"{self.reference_channel}_orientation_target.fits"
        )
        channel_target_file = os.path.join(
            self.work_base,
            f"{self.channel}_orientation_target.fits"
        )
        if not os.path.exists(reference_channel_target_file) and os.path.exists(channel_target_file):
            raise FileNotFoundError(
                f"No reference channel target file found at {reference_channel_target_file}. "
                f"However, a target file for the current working channel was found: {channel_target_file}. "
                "For coalignment through RosaZylaDestretch.register_rosa_zyla_observations, "
                f"the reference channel, {self.reference_channel} MUST be processed first."
            )
        if not os.path.exists(reference_channel_target_file):
            raise FileNotFoundError(f"File not found: {reference_channel_target_file}")
        if not os.path.exists(channel_target_file):
            raise FileNotFoundError(f"File not found: {channel_target_file}")

        offsets, rotation = align.align_derotate_channel_images(
            channel_target_file,
            reference_channel_target_file
        )
        self.theta += rotation
        # Since we're adjusting the CRPIX instead of actually applying a shift,
        # we need to flip the sign of the offsets.
        # e.g., if we need to move down and to the right, the reference pixel has
        # move up and to the left.
        self.offset = -offsets
        return

    def channel_apply_offsets(self) -> None:
        """
        Apply relative offsets and rotation between the
        working channel and the reference channel.
        This function is not called on the reference channel.
        """
        if self.pspkl_filelist == self.dstr_filelist == self.deflow_filelist == []:
            raise FileNotFoundError(
                f"No reduced files found in {self.work_base} for alignment."
            )
        full_filelist = self.pspkl_filelist + self.dstr_filelist + self.deflow_filelist
        for file in tqdm.tqdm(
            full_filelist,
            desc=f"Applying channel offsets relative to {self.reference_channel}",
            diable=not self.progress
        ):
            with fits.open(file, mode="update") as hdul:
                hdul[0].header["CRPIX1"] += self.offset[1] # image align returns yshift, xshift
                hdul[0].header["CRPIX2"] += self.offset[0] # FITS headers index xpixel, ypixel
                hdul[0].header["CROTA2"] += self.theta
                hdul.flush()
        return

    def update_reference_frame(self) -> None:
        """Performs simple alignment between the reference
        channel and the sun. By default, this is done to HMI
        continuum. If a different channel is required, it must
        be set manually.
        """
        if self.dstr_filelist != []:
            alignment_filelist = self.dstr_filelist
        else:
            alignment_filelist = self.pspkl_filelist
        try:
            reference_file = align.find_best_image_speckle_alpha(alignment_filelist)
        except KeyError:
            # No alpha value found in files. Use 0th file instead
            reference_file = alignment_filelist[0]
        refim, refhdr = align.read_rosa_zyla_image(reference_file)
        refmap = align.fetch_reference_image(refhdr, savepath=self.work_base, channel=self.master_alignment)
        updated_center = align.align_rotate_map(
            refim, refhdr, refmap
        )
        if self.verify_alignment:
            try:
                align.verify_alignment_accuracy(refim, updated_center, refmap)
            except align.PointingError:
                self.reference_coordinate = refhdr
            else:
                self.reference_coordinate = updated_center
        else:
            self.reference_coordinate = updated_center

        return

    def apply_solar_offsets(self) -> None:
        """Applies updated pointing information to all reduced filelists.
        """
        if self.pspkl_filelist == self.dstr_filelist == self.deflow_filelist == []:
            raise FileNotFoundError(
                f"No reduced files found in {self.work_base} for alignment."
            )
        full_filelist = self.pspkl_filelist + self.dstr_filelist + self.deflow_filelist
        if self.reference_coordinate == {}:
            warnings.warn(
                "Reference coordinate not given. "
                f"If the pointing offsets were applied to {self.reference_channel} previously, "
                "this should be fine. We'll try and pull the coordinate from "
                f"{self.reference_channel_working_directory}."
            )
            search = sorted(glob.glob(
                os.path.join(self.reference_channel_working_directory, "splineDestretch", "*.fits")
            ))
            if len(search) == 0:
                raise FileNotFoundError("Never mind. I didn't find any files.")
            _, self.reference_coordinate = align.read_rosa_zyla_image(search[0])
        align.update_imager_pointing_values(
            full_filelist, self.reference_coordinate,
            additional_rotation=self.theta, progress=self.progress
        )
        return

    def generate_context_movie(self) -> None:
        """Creates context movie from final data"""
        if self.pspkl_filelist == self.dstr_filelist == self.deflow_filelist == []:
            raise FileNotFoundError("No lists of reduced files in class!")
        if not os.path.exists(self.context_movie_directory):
            print(
                f"{__name__}: os.mkdir: attempting to create directory: {self.context_movie_directory}"
            )
            os.mkdir(self.context_movie_directory)
        if self.pspkl_filelist != []:
            movie_name = f"{self.channel}_postspeckle_{self.date}_{self.time}.mp4"
            movie.rosa_hardcam_movie_maker(
                self.pspkl_filelist, self.context_movie_directory,
                movie_name, self.channel, progress=self.progress
            )
        if self.dstr_filelist != []:
            movie_name = f"{self.channel}_destretch_{self.date}_{self.time}.mp4"
            movie.rosa_hardcam_movie_maker(
                self.dstr_filelist, self.context_movie_directory,
                movie_name, self.channel, progress=self.progress
            )
        if self.deflow_filelist != []:
            movie_name = f"{self.channel}_flowpreserved_destretch_{self.date}_{self.time}.mp4"
            movie.rosa_hardcam_movie_maker(
                self.deflow_filelist, self.context_movie_directory,
                movie_name, self.channel, progress=self.progress
            )

        raise
