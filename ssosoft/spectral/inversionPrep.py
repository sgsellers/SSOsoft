import configparser
import glob
import os
import re
import sys
import warnings
from importlib import resources
from importlib import util

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy
import numpy as np
import scipy.interpolate as scinterp

if util.find_spec("hazel") is not None:
    # Required if you want to write reference atmosphere files
    import hazel
import h5py
import shutil

from . import spectraTools as spex

class InversionPrep:
    """
    Handles additional processing steps for preparing 3D spectropolarimetric inversions
    from SSO data products. As inversions typically require high-performance computing
    resources, this class is not intended to actually perform the inversion process.

    Rather, this performs additional corrections and normalizations, then writes relevant
    files in the format expected by the inversion code selected (at the moment, just
    Hazel2 with SIR support). The user will then have to transfer the created directory
    and files to a compatible supercomputer, perform the inversions, and transfer the
    results back to their local machine. At this point, the class can be invoked once more
    in order to write the results back into FITS format.
    """

    def __init__(self, config_file: str) -> None:
        """

        Parameters
        ----------
        config_file : str

        """
        self.config_file = config_file

        # We'll set up some defaults for the case where we've got 6302, 8542, or 10830 being prepped.
        # We should avoid popping up widgets if we can at all avoid it.

        # Default range, chosen to select both lines with a bit of continuum on either side
        self.spectral_range_6302 = [6300.75, 6303.25]
        # Default range, chosen to select an even(ish) range around the line core avoiding
        # contamination from the Fe lines in the blue wing.
        self.spectral_range_8542 = [8538.75, 8545.45]
        # We'll be merging the FIRS inversions into this as well, might as well set it up now.
        # Note that, unlike with firs-tools, we'll be including the telluric line in the He I
        # red wing. We'll start including a telluric atmosphere for it, since during flares,
        # redshifts into that line are not uncommon.
        self.spectral_range_10830 = [10823.25, 10833.0]
        # Two ranges, just in case one falls outside the chip
        self.continuum_range_6302 = np.array(
            [[6296.75, 6297.5],
            [6304.5, 6305.1]]
        )
        # Typically, the wings of the line extend across the SPINOR chip.
        # For 8542 normalization, pick a couple ranges that at the edges,
        # and we'll get an additional fudge factor from the FTS atlas mean
        # across those regions.
        self.continuum_range_8542 = np.array(
            [[8532.5, 8534.5],
             [8545.5, 8547.75]]
        )
        # We should be conservative with the 10830 continuum range, just in case we ever
        # use the DWDM and a multi-slit unit.
        self.continuum_range_10830 = np.array(
            [[10823, 10824.25]]
        )

        # Default telluric line centers for 6302 and 10830.
        self.telluric_centers_6302 = np.array([6301.99, 6302.75])
        self.telluric_centers_10830 = np.array([10832.1])

        # For selecting quieter regions from a continuum image.
        # This can be settable for cases that aren't active regions
        self.percentiles = [50, 95]
        # Default IQUV boundaries
        self.boundary_default = np.array([1.25, 0., 0., 0.]) #!
        # If True, writes a reference photosphere and chromosphere
        self.write_atmos = False
        self.write_config = True

        self.range_method = ""
        self.inversion_code = ""
        self.inversion_directory = ""
        self.inversion_file_pattern = ""
        self.inversion_final_directory = ""
        self.central_wavelength = ""

        self.verbose = False #!!
        self.overview_plot = True #!!

        self.bin_slits = 1 #!!
        self.bin_spatial = 1 #!!

        return


    def write_inversion_files(self):
        """
        Master function for pre-inversion
        Returns
        -------

        """
        cameras, codes = self.parse_config_file()
        for camera, code in zip(cameras, codes):
            if self.verbose:
                print("Proceeding with processing of {0}".format(camera))
            map_list = self.parse_camera_params(camera)
            outdir_list = self.set_up_output_directories(len(map_list))
            self.inversion_code = code
            for map_file, index in zip(map_list, range(len(map_list))):
                ranges = self.get_inversion_ranges(map_file)
                stokes_norm, wave_grid, stokes_noise, coord_grid = self.prep_data(map_file, ranges)
                self.populate_files(
                    stokes_norm, wave_grid, stokes_noise, coord_grid, ranges, outdir_list[index], map_file
                )

        return


    def write_level2_files(self) -> None:
        """
        Master function for post-inversion.
        Does similar loop to "write_inversion_files", but grabs the post-inversion file and repacks it to fits.

        Returns
        -------
        None
        """
        cameras, codes = self.parse_config_file()
        for camera, code in zip(cameras, codes):
            map_list = self.parse_camera_params(camera)
            outdir_list = self.set_up_output_directories(len(map_list))
            for outdir, index in zip(outdir_list, range(len(outdir_list))):
                if code == "hazel":
                    filename = self.repack_inversion(outdir)
                    if self.overview_plot:
                        self.plot_inversion_overview(filename)


    def parse_config_file(self):
        """
        Parses the "INVERSIONS" section of the config file, returns a list of camera sections for further parsing

        Returns
        -------
        cameras : list
        codes : list
        """
        config = configparser.ConfigParser()
        config.read(self.config_file)

        self.inversion_directory = config['INVERSIONS']["inversionFileDirectory"]
        cameras = config['INVERSIONS']['cameras'].split(",")
        # set this up to take different codes per camera, even if other codes aren't implemented yet
        codes = config['INVERSIONS']['inversionCode'].split(",")
        if len(codes) < len(cameras):
            codes = codes * len(cameras)
        self.range_method = config['INVERSIONS']["spectralRange"]
        atmos = config['INVERSIONS']['writeAtmos'] if 'writeatmos' in config['INVERSIONS'].keys() else ""
        if atmos.lower() == "true":
            self.write_atmos = True
        elif atmos.lower() == "false":
            self.write_atmos = False
        cfig = config['INVERSIONS']['writeConfig'] if "writeconfig" in config["INVERSIONS"].keys() else ""
        if cfig.lower() == "true":
            self.write_config = True
        elif cfig.lower() == "false":
            self.write_config = False
        cbound = config['INVERSIONS']['boundaries'] if "boundaries" in config['INVERSIONS'].keys() else ""
        cbound = cbound.split(",")
        if len(cbound) == 4:
            self.boundary_default = np.array([float(i) for i in cbound])
        cpct = config['INVERSIONS']['continuumBound'] if "continuumbound" in config['INVERSIONS'].keys() else ""
        cpct = cpct.split(",")
        if len(cpct) == 2:
            self.percentiles = [float(i) for i in cpct]
        verb = config['INVERSIONS']['verbose'] if 'verbose' in config['INVERSIONS'].keys() else ""
        if verb.lower() == "true":
            self.verbose = True
        elif verb.lower() == "false":
            self.verbose = False
        oview = config['INVERSIONS']['overviewPlot'] if 'overviewplot' in config['INVERSIONS'].keys() else ""
        if oview.lower() == "true":
            self.overview_plot = True
        elif oview.lower() == "false":
            self.overview_plot = False

        self.bin_slits = int(
            config['INVERSIONS']['binSlits']
        ) if 'binslits' in config['INVERSIONS'].keys() else self.bin_slits
        self.bin_spatial = int(
            config['INVERSIONS']['binSpatial']
        ) if 'binspatial' in config['INVERSIONS'].keys() else self.bin_spatial

        return cameras, codes


    def parse_camera_params(self, camera_name):
        """
        Sets up channel-specific variable. Uses the "cameraName" section in the configfile
        Parameters
        ----------
        camera_name : str

        Returns
        -------
        map_list

        """
        config = configparser.ConfigParser()
        config.read(self.config_file)
        map_pattern = config[camera_name]['reducedFilePattern']
        # We'll do this the smart way, and use a regex pattern to sub all
        # string formatters with wildcards for easy globbing.
        pattern = "\\{[^}]*\\}" # Grabs curly brackets and everything in them
        map_pattern = re.sub(pattern, "*", map_pattern)
        map_dir = config[camera_name]['reducedFileDirectory']
        map_list = sorted(glob.glob(os.path.join(map_dir, map_pattern)))
        self.central_wavelength = config[camera_name]['centralWavelength']
        self.inversion_file_pattern = config[camera_name]['inversionFilePattern']
        self.inversion_final_directory = config[camera_name]['inversionFileDestination']

        return map_list


    def set_up_output_directories(self, nmaps: int) -> list:
        """
        Creates directory structure for output files to be placed in

        Parameters
        ----------
        nmaps : int
            Number of map files being prepped for inversions

        Returns
        -------
        outdir_list: list
            List of output directories

        """

        if not os.path.exists(self.inversion_directory):
            os.mkdir(self.inversion_directory)

        if not os.path.exists(os.path.join(self.inversion_directory, self.central_wavelength)):
            os.mkdir(os.path.join(self.inversion_directory, self.central_wavelength))

        outdir_list= []
        for i in range(nmaps):
            outdir_name = os.path.join(self.inversion_directory, self.central_wavelength, "map_{0}".format(i))
            os.mkdir(outdir_name) if not os.path.exists(outdir_name) else None
            outdir_list.append(outdir_name)

        return outdir_list


    def get_inversion_ranges(self, map_file: str) -> dict:
        """
        If self.rangeMethod is set to default, sets spectral, continuum, telluric ranges to the
        included defaults. If it's set to "choose", calls a series of widgets for the user to pick
        regions of interest.

        Parameters
        ----------
        map_file : str
            Map file to process for inversions

        Returns
        -------
        ranges : dict
            Dictionary of the spectral range, continuum range, telluric line centers

        """
        ranges = {
            "Spectral Range": [],
            "Continuum Range": [],
            "Telluric Centers": []
        }
        # Try to set default ranges first:
        if (self.range_method == "default") & any([self.central_wavelength == x for x in ["6302", "8542", "10830"]]):
            if self.central_wavelength == "6302":
                ranges["Spectral Range"] =  self.spectral_range_6302
                ranges["Continuum Range"] = self.continuum_range_6302
                ranges["Telluric Centers"] = self.telluric_centers_6302
            elif self.central_wavelength == "8542":
                ranges["Spectral Range"] = self.spectral_range_8542
                ranges["Continuum Range"] = self.continuum_range_8542
                ranges["Telluric Centers"] = []
            elif self.central_wavelength == "10830":
                ranges["Spectral Range"] = self.spectral_range_10830
                ranges["Continuum Range"] = self.continuum_range_10830
                ranges["Telluric Centers"] = self.telluric_centers_10830

        else:
            reference_profile, wavelength_grid = self.get_reference_profile(map_file)
            spectral_range, continuum_range, telluric_centers = self.inversion_picker(reference_profile, wavelength_grid)
            ranges["Spectral Range"] = spectral_range
            ranges["Continuum Range"] = continuum_range
            ranges["Telluric Centers"] = telluric_centers

        return ranges


    def prep_data(
            self, map_file: str, spectral_ranges: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs prep and normalization of Stokes data.
        Assembles coordinate grid, normalizes profiles to disk-center quiet-sun intensity,
        estimates noise from std.dev. of continuum regions, and clips to the region of interest.

        Parameters
        ----------
        map_file : str
            Path to the reduced raster file
        spectral_ranges : dict
            dictionary contiaining spectral ranges

        Returns
        -------
        stokes_norm : numpy.ndarray
            Stokes profiles, normalized to disk-center quiet sun. Shape (4, ny, nx, nlambda)
        wave_grid : numpy.ndarray
            Corresponding wavelength grid. Shape (nlambda, )
        stokes_noise : numpy.ndarray
            Noise estimates for Stokes profiles. Shape (4, ny, nx, nlambda).
            Only one noise value per pixel per Stokes vector, but it's repeated along the nlambda axis
        coord_grid : numpy.ndarray
            Hazel-compatible coordinate grid. Shape (4, ny, nx) for theta, phi, gamma
        """
        if self.verbose:
            print("Loading data from:\n{0}".format(map_file))
        with fits.open(map_file) as hdul:
            # Assemble stokes profiles
            stokes_norm = np.zeros((4, *hdul['STOKES-I'].data.shape))
            l = ["I", "Q", "U", "V"]
            for i in range(4):
                stokes_norm[i] = hdul['STOKES-{0}'.format(l[i])].data
            # grab wavelength array
            wavelength_array = hdul['lambda-coordinate'].data
            # grab coordinate values for setting up the grid
            xcen = hdul[0].header['XCEN']
            ycen = hdul[0].header['YCEN']
            fovx = hdul[0].header['FOVX']
            fovy = hdul[0].header['FOVY']
            rotation = hdul[0].header['ROT']
            dx = hdul['STOKES-I'].header['CDELT1']
            dy = hdul['STOKES-I'].header['CDELT2']
            # Edge case where FOV is 0 in either direction
            fovx = dx if fovx == 0 else fovx
            fovy = dy if fovy == 0 else fovy
            # Try to grab hairlines from file headers
            if "HAIRLIN0" in hdul[0].header.keys():
                if "HAIRLIN1" in hdul[0].header.keys():
                    hairline_centers = (hdul[0].header['HAIRLIN0'], hdul[0].header['HAIRLIN1'])
                    # Edge case: hairline centers incorrectly registered too close.
                    if np.diff(hairline_centers)[0] < 50:
                        hairline_centers = tuple()
                else:
                    hairline_centers = tuple([hdul[0].header['HAIRLIN0']])
            else:
                hairline_centers = tuple()

        if self.verbose:
            print("Assembling coordinate grid")
        # Start by assembling the coordinate grid:
        coord_grid = self.assemble_coord_grid(
            xcen, ycen, fovx, fovy, dx, dy, rotation, (stokes_norm.shape[1], stokes_norm.shape[2]) # Y/X shape
        )
        # Need Hazel installed for this part.
        if self.verbose:
            print("Correcting for limb darkening")
        clv_factor = self.center_to_limb_variation(coord_grid[0], wavelength_array)
        # Assemble 2D image of Stokes-I in the continuum. Use this to make a mask of quiet(er) regions.
        # From this and the defined continuum windows, we can get a quiet-sun continuum value at this
        # projection angle.

        # Case where there are two hairlines
        if len(hairline_centers) == 2:
            continuum_determination_region = stokes_norm[
                0, int(min(hairline_centers)) + 5:int(max(hairline_centers)) - 5, :, :
            ]
        else:
            # Cut the edge 100 pixels.
            continuum_determination_region = stokes_norm[
                0, 50: -50, :, :
            ]
        if self.verbose:
            print("Determining local continuum")
        continuum_value = self.determine_local_continuum_intensity(
            continuum_determination_region, spectral_ranges['Continuum Range'], wavelength_array
        )

        if self.bin_spatial != 1:
            stokes_norm = self._bin_array(stokes_norm, 1, self.bin_spatial, np.nansum)
            coord_grid = self._bin_array(coord_grid, 1, self.bin_spatial, np.nanmean)
            clv_factor = self._bin_array(clv_factor, 1, self.bin_spatial, np.nanmean)
            continuum_value *= self.bin_spatial
        if self.bin_slits != 1:
            stokes_norm = self._bin_array(stokes_norm, 2, self.bin_slits, np.nansum)
            coord_grid = self._bin_array(coord_grid, 2, self.bin_slits, np.nanmean)
            clv_factor = self._bin_array(clv_factor, 2, self.bin_slits, np.nanmean)
            continuum_value *= self.bin_slits

        # Multiple Stokes profiles by this to normalize
        disk_center_quiet_sun_factor = clv_factor / continuum_value
        stokes_norm *= np.repeat(disk_center_quiet_sun_factor[np.newaxis, :, :, :], 4, axis=0)
        # Grab noise regions from continuum ranges
        stokes_noise = np.zeros(stokes_norm.shape)
        for i in range(stokes_norm.shape[0]):
            noise_maps = np.zeros((spectral_ranges['Continuum Range'].shape[0], *stokes_norm.shape[1:3]))
            for j in range(spectral_ranges['Continuum Range'].shape[0]):
                indices = sorted(
                    [int(spex.find_nearest(wavelength_array, k)) for k in spectral_ranges['Continuum Range'][j]]
                )
                noise_maps[j] = np.nanstd(stokes_norm[i, :, :, indices[0]:indices[1]], axis=-1)
            stokes_noise[i] = np.repeat(
                np.nanmean(noise_maps, axis=0)[:, :, np.newaxis], stokes_noise.shape[3], axis=-1
            )
        # Clip normalized Stokes Profiles, noise values, and wavelength arrays to the selected spectral ranges
        indices = sorted(
            [int(spex.find_nearest(wavelength_array, i)) for i in spectral_ranges['Spectral Range']]
        )

        stokes_norm = stokes_norm[:, :, :, indices[0]:indices[1]]
        stokes_noise = stokes_noise[:, :, :, indices[0]:indices[1]]
        wave_grid = wavelength_array[indices[0]:indices[1]]

        return stokes_norm, wave_grid, stokes_noise, coord_grid

    def determine_local_continuum_intensity(
            self, intensity_spectra: np.array, continuum_ranges: np.array, wavelength_array: np.array
    ) -> np.float64:
        """
        Determines mean of local quiet-sun continuum intensity.
        Forms an image of the mean continuum from the ranges provided, then masks out all pixels
        outside some percentile values. Default is 50th and 95th percentile.
        In an active region raster, this does a pretty decent job of providing a population
        of pixels that aren't umbral or penumbral, while also avoiding overly bright values
        (usually caused by rotating modulator desync. You'll see this in columns that are
        overly bright compared to their neighbors.)

        Parameters
        ----------
        intensity_spectra : numpy.ndarray
            3D raster image of shape (ny, nx, nlambda)
        continuum_ranges : numpy.ndarray
            ND array of ranges corresponding to continuum sections of spectrum. Shape (nregions, 2)
        wavelength_array : numpy.ndarray
            1D array of continuum values

        Returns
        -------
        mean_continuum : float

        """
        mean_continuum = np.zeros(continuum_ranges.shape[0])
        for i in range(continuum_ranges.shape[0]):
            indices = sorted([int(spex.find_nearest(wavelength_array, j)) for j in continuum_ranges[i]])
            continuum_image = np.nanmean(intensity_spectra[:, :, indices[0]:indices[1]], axis=-1)
            percentile_values = np.percentile(continuum_image, self.percentiles)
            continuum_image[continuum_image < percentile_values[0]] = np.nan
            continuum_image[continuum_image > percentile_values[1]] = np.nan
            mean_continuum[i] = np.nanmean(continuum_image)
        mean_continuum = np.nanmean(mean_continuum)

        return mean_continuum

    def populate_files(
            self,
            stokes_norm: np.ndarray, wave_grid: np.ndarray, stokes_noise: np.ndarray, coord_grid: np.ndarray,
            ranges: dict, outdir: str, map_file: str
    ) -> None:
        """
        Saves pre-inversion files to directory created for the purpose.

        Parameters
        ----------
        stokes_norm : numpy.ndarray
            Unflattened array of Stokes vectors. Shape (4, ny, nx, nlambda)
        wave_grid : numpy.ndarray
            1D array of wavelength values along the grid. Shape (nlambda, )
        stokes_noise : numpy.ndarray
            Unflattened array of noise values. Shape (4, ny, nx, nlambda)
            Technically, it's (4, ny, nx), then repeated along nlambda
        coord_grid : numpy.ndarray
            Coordinate grid for inversions. Shape (3, ny, nx)
        ranges : dict
            Contains values for Telluric line centers. Used in creating atmospheres.
        outdir : str
            Path to save output files to.
        map_file : str
            Path to input fits file

        Returns
        -------
        None

        """
        # Flatten Stokes vectors and noise to (4, npixels, nlambda)
        npix = stokes_norm.shape[1] * stokes_norm.shape[2]
        stokes_norm = stokes_norm.reshape(4, npix, stokes_norm.shape[3])
        stokes_noise = stokes_noise.reshape(4, npix, stokes_noise.shape[3])
        coord_grid = coord_grid.reshape(3, npix)
        # Move axes to (npixels, nlambda, 4)
        stokes_norm = np.moveaxis(stokes_norm, 0, 2)
        stokes_noise = np.moveaxis(stokes_noise, 0, 2)
        coord_grid = np.swapaxes(coord_grid, 0, 1)

        boundary = np.repeat(
            self.boundary_default[np.newaxis, np.newaxis, :], stokes_norm.shape[1], axis=1
        )
        boundary = np.repeat(boundary, npix, axis=0)

        # Main h5-formatted file with stokes profiles, noise values, coordinate grid, boundary conditions
        with h5py.File(os.path.join(outdir, "{0}_preinversion.h5".format(self.inversion_code)), mode="w") as input_file:
            stokes_db = input_file.create_dataset('stokes', stokes_norm.shape, dtype=np.float64)
            stokes_db[:] = np.nan_to_num(stokes_norm)
            noise_db = input_file.create_dataset('sigma', stokes_noise.shape, dtype=np.float64)
            stokes_noise = np.nan_to_num(stokes_noise)
            stokes_noise[stokes_noise == 0] = np.nanmedian(stokes_noise[stokes_noise != 0])
            noise_db[:] = stokes_noise
            los_db = input_file.create_dataset("LOS", coord_grid.shape, dtype=np.float64)
            los_db[:] = np.nan_to_num(coord_grid, nan=90)
            boundary_db = input_file.create_dataset("boundary", stokes_norm.shape, dtype=np.float64)
            boundary_db[:] = boundary
        if self.verbose:
            print("Pre-inversion data file written in:\n{0}".format(outdir))
        # Wavelength grid as plaintext
        np.savetxt(os.path.join(outdir, "hazel_preinversion.wavelength"), wave_grid, header='lambda')
        # Weight file as plaintext
        with open(os.path.join(outdir, "hazel_preinversion.weights"), "w") as weights:
            weights.write('# WeightI WeightQ WeightU WeightV\n')
            for i in range(stokes_norm.shape[1]):
                weights.write('1.0    1.0    1.0    1.0\n')
        # Reference atmospheres
        if self.write_atmos & ("hazel" in sys.modules):
            photosphere = hazel.tools.File_photosphere(mode="multi")
            photosphere.set_default(n_pixel=npix, default="hsra")
            photosphere.save(os.path.join(outdir, "reference_photosphere"))
            # Need to set on/off disk chromospheres.
            chromosphere = hazel.tools.File_chromosphere(mode='multi')
            chromosphere.set_default(n_pixel=npix, default='disk')
            offlimb = hazel.tools.File_chromosphere(mode='multi')
            offlimb.set_default(n_pixel=1, default='offlimb')
            for i in range(coord_grid.shape[0]):
                if np.isnan(coord_grid[i, 0]):
                    chromosphere.model['model'][i, :] = offlimb.model['model'][0, :]
            chromosphere.save(os.path.join(outdir, 'reference_chromosphere'))
            if self.verbose:
                print("Reference Photosphere and Chromosphere files written in:\n{0}".format(outdir))
        if self.write_atmos and not ("hazel" in sys.modules):
            warnings.warn("Hazel is not installed in the current environment. Atmosphere files cannot be written.")
        # Write telluric atmospheres:
        for i in range(len(ranges['Telluric Centers'])):
            with open(os.path.join(outdir, "telluric_{0}.1d".format(i)), "w") as telluric:
                telluric.write("lambda0\tsigma\tdepth\ta\tff\n")
                telluric.write("{:0.2f}\t0.1\t0.2\t1.0\t1.0".format(
                    ranges['Telluric Centers'][i]
                ))
            if self.verbose:
                print("Telluric Line File with Lambda0 {0:0.2f} written to:\n{1}".format(
                    ranges['Telluric Centers'][i], os.path.join(outdir, "telluric_{0}.1d".format(i))
                ))
        # Default config file, inversion file
        if self.write_config:
            if (self.central_wavelength == "6302") & (self.inversion_code == "hazel"):
                with resources.path('ssosoft.spectral.inversions', 'config6302Default.ini') as rpath:
                    cfile = shutil.copy(rpath, os.path.join(outdir, "config6302.ini"))
                if self.verbose:
                    print("Default configuration file written to:\n{0}".format(cfile))
            elif (self.central_wavelength == "8542") & (self.inversion_code == "hazel"):
                with resources.path("ssosoft.spectral.inversions", "config8542Default.ini") as rpath:
                    cfile = shutil.copy(rpath, os.path.join(outdir, "config8542.ini"))
                if self.verbose:
                    print("Default configuration file written to:\n{0}".format(cfile))
            elif (self.central_wavelength == "10830") & (self.inversion_code == "hazel"):
                with resources.path("ssosoft.spectral.inversions", "config10830Default.ini") as rpath:
                    cfile = shutil.copy(rpath, os.path.join(outdir, "config10830.ini"))
                if self.verbose:
                    print("Default configuration file written to:\n{0}".format(cfile))
            elif self.inversion_code == "hazel":
                with resources.path("ssosoft.spectral.inversions", "config10830Default.ini") as rpath:
                    cfile = shutil.copy(rpath, os.path.join(outdir, "config.ini"))
                if self.verbose:
                    print("Default configuration file written to:\n{0}".format(cfile))

            # Write a hazel invert.py file. Don't like it, but it isn't for me...
            if self.inversion_code == "hazel":
                with open(os.path.join(outdir, "invert.py"), "w") as invert_file:
                    invert_file.write("import hazel\n\n")
                    invert_file.write("iterator = hazel.Iterator(use_mpi=True)\n")
                    invert_file.write(
                        "mod = hazel.Model(\'{0}\', rank=iterator.get_rank(), working_mode=\'inversion\', verbose=4)\n".format(
                            os.path.split(cfile)[1]
                        )
                    )
                    invert_file.write(
                        "iterator.use_model(model=mod)\n"
                    )
                    invert_file.write(
                        "iterator.run_all_pixels()"
                    )
                if self.verbose:
                    print("Default inversion script written to:\n{0}".format(
                        os.path.join(outdir, "invert.py")
                    ))
        # Last, but not least, write a text file with the filepaths, bin factor
        with open(os.path.join(outdir, "prep_parameters.txt"), "w") as prep:
            prep.write(
                "init_file={0}\n".format(
                    "{0}_preinversion.h5".format(self.inversion_code)
                )
            )
            prep.write(
                "fits_file={0}\n".format(
                    map_file
                )
            )
            prep.write(
                "bin_slits={0}\n".format(
                    self.bin_slits
                )
            )
            prep.write(
                "bin_spatial={0}\n".format(
                    self.bin_spatial
                )
            )

        return None

    def repack_inversion(
            self, inversion_dir: str,
            inverted_file: str=None,
            spectrum_key: str=None, chromosphere_key: str or list='ch1', photosphere_key: str='ph1'
    ) -> str:
        """
        Repacks inverted h5-type file into
        Parameters
        ----------
        inversion_dir : str
        inverted_file : str or None, optional
            If set, should be the path to the post-inversion file.
            If it isn't set, the code tries to find it by checking odd-man-out filenames
        spectrum_key : str or None, optional
            If set, should point to the key in the final h5 file corresponding to the spectrum
            Otherwise, the code guesses it
        chromosphere_key : str or None, optional
            If set, should point to the key in the final h5 file, corresponding to the chromospheric params
            Code will guess if not set. Not all inversions have chromospheres.
            If it's a list, each entry should correspond to one chromosphere
        photosphere_key: str or None, optional.
            If set, should point to the key in the final h5 file, corresponding to the photospheric params
            Otherwise, the code guesses it.

        Returns
        -------
        outfile : str
            Name of final file

        """
        with open(os.path.join(inversion_dir, "prep_parameters.txt"), "r") as prep:
            lines = prep.readlines()
            initial_file = lines[0].split("=")[1].replace("\n", "")
            fits_file = lines[1].split("=")[1].replace("\n", "")
            bin_slits = int(lines[2].split("=")[1].replace("\n", ""))
            bin_spatial = int(lines[3].split("=")[1].replace("\n", ""))
        if inverted_file is None:
            candidate_list = sorted(glob.glob(os.path.join(inversion_dir, "*.h5")))
            known_slugs = ['reference_chromosphere', 'reference_photosphere', os.path.split(initial_file)[1]]
            inverted_file = [
                i for i in candidate_list if not any([k in i for k in known_slugs])
            ][0]
        with h5py.File(os.path.join(inversion_dir, initial_file), "r") as init:
            preinversion_stokes = init['stokes'][:, :, :]
        if spectrum_key is None and self.central_wavelength == "6302":
            spectrum_key = "spec6302"
        elif spectrum_key is None and self.central_wavelength == "8542":
            spectrum_key = "spec8542"
        elif spectrum_key is None and self.central_wavelength == "10830":
            spectrum_key = "spec10830"
        with h5py.File(inverted_file, "r") as invert:
            if spectrum_key is None:
                # Make an intelligent guess.
                keys = invert.keys()
                spectrum_key = [i for i in keys if "spec" in i][0]
            wave_grid = invert[spectrum_key]["wavelength"][:]
        # Grab what we'll need to write our headers in the output file.
        with fits.open(fits_file) as hdul:
            dx = hdul['STOKES-I'].header['CDELT1'] * bin_slits
            dy = hdul['STOKES-I'].header['CDELT2'] * bin_spatial
            nx = int((hdul['STOKES-I'].header['NAXIS3'] - 1) / bin_spatial) + 1
            ny = int((hdul['STOKES-I'].header['NAXIS2'] - 1) / bin_slits) + 1
            dlambda = hdul['STOKES-I'].header['CDELT3']
            crval1 = hdul['STOKES-I'].header['CRVAL1']
            crval2 = hdul['STOKES-I'].header['CRVAL2']
            crota2 = hdul['STOKES-I'].header['CROTA2']
            hdul_names = [hdul[i].header['EXTNAME'] for i in range(1, len(hdul))]
            if "METADATA" in hdul_names:
                metadata = hdul['METADATA'].data.copy()
            else:
                metadata = None
            master_header = hdul[0].header.copy()

        # Start date/time:
        date, time = master_header['STARTOBS'].split("T")
        date = date.replace('-', '')
        time = str(round(float(time.replace(":", "")), 0)).split(".")[0]
        outname = self.inversion_file_pattern.format(
            date,
            time,
            self.inversion_code
        )
        outfile = os.path.join(self.inversion_final_directory, outname)

        # Relevant parameters...
        chromosphere_params = [
            'Bx', 'Bx_err',
            'By', 'By_err',
            'Bz', 'Bz_err',
            'v', 'v_err',
            'deltav', 'deltav_err',
            'tau', 'tau_err',
            'a', 'a_err',
            'beta', 'beta_err',
            'ff', 'ff_err'
        ]
        chromosphere_units = [
            'Gauss', 'Gauss',
            'Gauss', 'Gauss',
            'Gauss', 'Gauss',
            'km/s', 'km/s',
            'km/s', 'km/s',
            'log10(OpticalDepth)', 'log10(OpticalDepth)',
            'Damping', 'Damping',
            'plasmaB', 'plasmaB',
            'FillFactor', 'FillFactor'
        ]
        photosphere_params = [
            'Bx', 'Bx_err',
            'By', 'By_err',
            'Bz', 'Bz_err',
            'T', 'T_err',
            'v', 'v_err',
            'vmac', 'vmac_err',
            'vmic', 'vmic_err',
            'ff', 'ff_err'
        ]
        photosphere_units = [
            'Gauss', 'Gauss',
            'Gauss', 'Gauss',
            'Gauss', 'Gauss',
            'Kelvin', 'Kelvin',
            'km/s', 'km/s',
            'km/s', 'km/s',
            'km/s', 'km/s',
            'FillFactor', 'FillFactor'
        ]

        # Eventual hdulist
        fits_hdulist = []
        # Set up primary HDU with relevant header values
        ext0 = fits.PrimaryHDU()
        ext0.header = master_header
        ext0.header['DATE'] = (np.datetime64('now').astype(str), "File creation date and time")
        ext0.header['DATA_LEV'] = 2
        del ext0.header['BTYPE']
        del ext0.header['BUNIT']
        if bin_slits != 1:
            ext0.header.insert(
                "NSUMEXP",
                ("NSUMSLIT", bin_slits, "Number of slit positions binned in inversion"),
                after=True
            )
        if bin_spatial != 1:
            ext0.header.insert(
                "NSUMEXP",
                ("NSUMSPAT", bin_spatial, "Bin factor along the slit for inversion"),
                after=True
            )
        nprsteps = len([i for i in ext0.header.keys() if "PRSETP" in i])
        ext0.header['PRSTEP{0}'.format(nprsteps + 1)] = ("INVERSION", self.inversion_code)
        fits_hdulist.append(ext0)

        with h5py.File(inverted_file, "r") as invert:
            # Pack any chromospheres first:
            if type(chromosphere_key) is str:
                chromosphere_key = [chromosphere_key]
            for key, ctr in zip(chromosphere_key, range(len(chromosphere_key))):
                if key in invert.keys():
                    chromosphere = invert[key]
                    columns = []
                    for param, unit in zip(chromosphere_params, chromosphere_units):
                        if "err" in param:
                            param_array = np.zeros((nx, ny))
                            error_array = chromosphere[param][:, 0, -1].reshape(nx, ny)
                            for x in range(error_array.shape[0]):
                                for y in range(error_array.shape[1]):
                                    if len(error_array) != 0:
                                        param_array[x, y] = error_array[x, y]
                        else:
                            param_array = chromosphere[param][:, 0, -1, 0].reshape(nx, ny)
                        columns.append(
                            fits.Column(
                                name=param,
                                format=str(int(nx*ny)) + "D",
                                dim='(' + str(param_array.shape[1]) + ','+str(param_array.shape[0]) + ')',
                                unit=unit,
                                array=param_array[np.newaxis, :, :]
                            )
                        )
                    ext = fits.BinTableHDU.from_columns(columns)
                    ext.header['EXTNAME'] = ("CHROMOSPHERE-{0}".format(ctr), "Fit chromospheric params from Hazel2")
                    ext.header['LINE'] = self.central_wavelength
                    ext.header["RSUN_ARC"] = master_header['RSUN_ARC']
                    ext.header['CDELT1'] = (dx, "arcsec")
                    ext.header['CDELT2'] = (dy, "arcsec")
                    ext.header['CTYPE1'] = 'HPLN-TAN'
                    ext.header['CTYPE2'] = 'HPLT-TAN'
                    ext.header['CUNIT1'] = 'arcsec'
                    ext.header['CUNIT2'] = 'arcsec'
                    ext.header['CRVAL1'] = (crval1, "Solar-X, arcsec")
                    ext.header['CRVAL2'] = (crval2, "Solar-Y, arcsec")
                    ext.header['CRPIX1'] = nx/2
                    ext.header['CRPIX2'] = ny/2
                    ext.header['CROTA2'] = (crota2, "degrees")
                    fits_hdulist.append(ext)

            # Write the FITS file now. As we move forward, we'll be using a bunch of memory.
            # We'll write the file, and then add extensions to it as we go.
            hdulist = fits.HDUList(fits_hdulist)
            hdulist.writeto(outname, overwrite=True)

            # Pack the photospheric params next. These have a height grid as well.
            photosphere = invert[photosphere_key]
            columns = []
            log_tau = photosphere['log_tau'][:]
            columns.append(
                fits.Column(
                    name='logTau',
                    format='D',
                    unit='Optical Depth',
                    array=log_tau
                )
            )
            for param, unit in zip(photosphere_params, photosphere_units):
                """ Slight explanation of the following code:
                Errors and fits are not straightforward.
                There are several ways the errors are recorded:
                    1.) Multiple nodes are fit. The error is an object array of errors at each fit node.
                        e.g., 5 nodes fit, the error array is an array of shape nx, ny. 
                        Each element of the error array is either:
                            ~A zero length array (could not fit the pixel)
                            ~An array of errors with a length equal to the number of nodes.
                    2.) A single node is fit. This is the simplest example of case 1 above.
                        Here, each element is either length zero or one.
                    3.) The parameter is not fit. Here, when the fit succeeds, the element is a 1-array with a nan.
                        This is vexatious.  
                Fit parameters can have nans in their nodelist AND empties, which must be accounted for as well
                vmac in the photosphere also doesn't have a height profile.
                
                2025-05-07: Okay, at some point Hazel+SIR changed its file outputs.
                As of right now, all of the "_nodes" keywords are just empty.
                Which is a problem when you put a while loop in your code.
                But the "_err" seems to now have the correct length each time, so we're going to go with that.
                """
                if ('vmac' in param) or ('ff' in param):
                    if len(photosphere[param][0, 0, -1]) == 0:
                        fill = np.zeros((len(log_tau), nx, ny))
                        columns.append(
                            fits.Column(
                                name=param,
                                format=str(int(nx * ny)) + "I",
                                dim='(' + str(fill.shape[2]) + "," + str(fill.shape[1]) + ")",
                                unit=unit,
                                array=fill
                            )
                        )
                    else:
                        # vmac is a special case -- no nodes
                        dummy_arr = np.zeros((len(log_tau), nx, ny))
                        if 'err' in param:
                            param_array = photosphere[param][:, 0, -1].reshape(nx, ny)
                            for x in range(param_array.shape[0]):
                                for y in range(param_array.shape[1]):
                                    if len(err[x, y]) != 0:
                                        dummy_arr[:, x, y] += param_array[x, y][0]
                        else:
                            param_array = photosphere[param][:, 0, -1, 0].reshape(nx, ny)
                            dummy_arr = np.repeat(param_array[np.newaxis, :, :], dummy_arr.shape[0], axis=0)
                        columns.append(
                            fits.Column(
                                name=param,
                                format=str(int(nx * ny)) + "D",
                                dim='(' + str(dummy_arr.shape[2]) + "," + str(dummy_arr.shape[1]) + ")",
                                unit=unit,
                                array=dummy_arr
                            )
                        )
                elif "err" in param:
                    # Case: Parameter not fit
                    if len(photosphere[param][0, 0, -1]) == 0:
                        fill = np.zeros((len(log_tau), nx, ny))
                        columns.append(
                            fits.Column(
                                name=param,
                                format=str(int(nx * ny)) + "I",
                                dim='(' + str(fill.shape[2]) + "," + str(fill.shape[1]) + ")",
                                unit=unit,
                                array=fill
                            )
                        )
                    elif len(photosphere[param][0, 0, -1] == 1):
                        # Case: parameter fit with one node. Cast error across tau grid
                        dummy_err = np.zeros((len(log_tau), nx, ny))
                        err = photosphere[param][:, 0, -1].reshape(nx, ny)
                        for x in range(err.shape[0]):
                            for y in range(err.shape[1]):
                                if len(err[x, y]) != 0:
                                    dummy_err[:, x, y] += err[x, y][0]
                        columns.append(
                            fits.Column(
                                name=param,
                                format=str(int(nx * ny)) + "D",
                                dim='(' + str(dummy_err.shape[2]) + "," + str(dummy_err.shape[1]) + ")",
                                unit=unit,
                                array=dummy_err
                            )
                        )
                    else:
                        # Case: parameter fit with multiple nodes. Interpolate along tau grid
                        # Would be way easier if the "_nodes" parameter existed to tell me exactly
                        # where the nodes *were*. Instead, we'll just do a linear interpolation
                        dummy_err = np.zeros((len(log_tau), nx, ny))
                        err = photosphere[param][:, 0, -1].reshape(nx, ny)
                        for x in range(err.shape[0]):
                            for y in range(err.shape[1]):
                                dummy_err[:, x, y] = np.interp(
                                    np.linspace(0, 1, num=len(log_tau)),
                                    np.linspace(0, 1, num=len(err[x, y])),
                                    err[x, y]
                                )
                        columns.append(
                            fits.Column(
                                name=param,
                                format=str(int(nx * ny)) + "D",
                                dim='(' + str(dummy_err.shape[2]) + "," + str(dummy_err.shape[1]) + ")",
                                unit=unit,
                                array=dummy_err
                            )
                        )
                else:
                    # Regular parameters
                    colarr = photosphere[param][:, 0, -1, :].reshape(nx, ny, len(log_tau))
                    colarr = np.transpose(colarr, (2, 0, 1))
                    columns.append(
                        fits.Column(
                            name=param,
                            format=str(int(nx * ny)) + "D",
                            dim='(' + str(colarr.shape[2]) + "," + str(colarr.shape[1]) + ")",
                            unit=unit,
                            array=colarr
                        )
                    )
            # Okay, we're back.
            ext = fits.BinTableHDU.from_columns(columns)
            ext.header['EXTNAME'] = ('PHOTOSPHERE', self.inversion_code)
            ext.header['LINE'] = self.central_wavelength
            ext.header["RSUN_ARC"] = master_header['RSUN_ARC']
            ext.header['CDELT1'] = (dx, "arcsec")
            ext.header['CDELT2'] = (dy, "arcsec")
            ext.header['CTYPE1'] = 'HPLN-TAN'
            ext.header['CTYPE2'] = 'HPLT-TAN'
            ext.header['CUNIT1'] = 'arcsec'
            ext.header['CUNIT2'] = 'arcsec'
            ext.header['CRVAL1'] = (crval1, "Solar-X, arcsec")
            ext.header['CRVAL2'] = (crval2, "Solar-Y, arcsec")
            ext.header['CRPIX1'] = nx / 2
            ext.header['CRPIX2'] = ny / 2
            ext.header['CROTA2'] = (crota2, "degrees")

            fits.append(outfile, ext.data, ext.header)

            # For completeness, we should include the pre- and post-inversion Stokes profiles.
            # Helpful for anyone checking their work.
            names = ['I', 'Q', 'U', 'V']
            for i in range(4):
                ext = fits.ImageHDU(preinversion_stokes[:, :, i].reshape(nx, ny, preinversion_stokes.shape[1]))
                ext.header['EXTNAME'] = (
                    "Stokes-{0}/Ic", "Normalized by Quiet Sun, Corrected for position angle".format(names[i])
                )
                ext.header["RSUN_ARC"] = master_header['RSUN_ARC']
                ext.header['CDELT1'] = (dx, "arcsec")
                ext.header['CDELT2'] = (dy, "arcsec")
                ext.header['CDELT3'] = dlambda
                ext.header['CTYPE1'] = 'HPLN-TAN'
                ext.header['CTYPE2'] = 'HPLT-TAN'
                ext.header['CTYPE3'] = "WAVE"
                ext.header['CUNIT1'] = 'arcsec'
                ext.header['CUNIT2'] = 'arcsec'
                ext.header['CUNIT3'] = 'Angstrom'
                ext.header['CRVAL1'] = (crval1, "Solar-X, arcsec")
                ext.header['CRVAL2'] = (crval2, "Solar-Y, arcsec")
                ext.header['CRVAL3'] = round(wave_grid[0], 3)
                ext.header['CRPIX1'] = nx / 2
                ext.header['CRPIX2'] = ny / 2
                ext.header['CRPIX3'] = 1
                ext.header['CROTA2'] = (crota2, "degrees")
                fits.append(outfile, ext.data, ext.header)
            # Synthetic Profiles
            for i in range(4):
                synth = invert[spectrum_key]['stokes'][:, 0, -1, i, :]
                synth = synth.reshape(nx, ny, synth.shape[1])
                ext = fits.ImageHDU(synth)
                ext.header['EXTNAME'] = (
                    "SYNTHStokes-{0}/Ic", "Normalized by Quiet Sun, Corrected for position angle".format(names[i])
                )
                ext.header["RSUN_ARC"] = master_header['RSUN_ARC']
                ext.header['CDELT1'] = (dx, "arcsec")
                ext.header['CDELT2'] = (dy, "arcsec")
                ext.header['CDELT3'] = dlambda
                ext.header['CTYPE1'] = 'HPLN-TAN'
                ext.header['CTYPE2'] = 'HPLT-TAN'
                ext.header['CTYPE3'] = "WAVE"
                ext.header['CUNIT1'] = 'arcsec'
                ext.header['CUNIT2'] = 'arcsec'
                ext.header['CUNIT3'] = 'Angstrom'
                ext.header['CRVAL1'] = (crval1, "Solar-X, arcsec")
                ext.header['CRVAL2'] = (crval2, "Solar-Y, arcsec")
                ext.header['CRVAL3'] = round(wave_grid[0], 3)
                ext.header['CRPIX1'] = nx / 2
                ext.header['CRPIX2'] = ny / 2
                ext.header['CRPIX3'] = 1
                ext.header['CROTA2'] = (crota2, "degrees")
                fits.append(outfile, ext.data, ext.header)
            # Chi-squared map:
            chi2 = invert[spectrum_key]['chi2'][:, 0, -1].reshape(nx, ny)
            ext = fits.ImageHDU(chi2)
            ext.header['EXTNAME'] = (
                "CHISQ", "Fit chi-squared calue"
            )
            ext.header["RSUN_ARC"] = master_header['RSUN_ARC']
            ext.header['CDELT1'] = (dx, "arcsec")
            ext.header['CDELT2'] = (dy, "arcsec")
            ext.header['CTYPE1'] = 'HPLN-TAN'
            ext.header['CTYPE2'] = 'HPLT-TAN'
            ext.header['CUNIT1'] = 'arcsec'
            ext.header['CUNIT2'] = 'arcsec'
            ext.header['CRVAL1'] = (crval1, "Solar-X, arcsec")
            ext.header['CRVAL2'] = (crval2, "Solar-Y, arcsec")
            ext.header['CRPIX1'] = nx / 2
            ext.header['CRPIX2'] = ny / 2
            ext.header['CROTA2'] = (crota2, "degrees")
            fits.append(outfile, ext.data, ext.header)

            # Wavelength array:
            ext = fits.ImageHDU(wave_grid)
            ext.header['EXTNAME'] = 'lambda-coordinate'
            ext.header['CTYPE'] = 'lambda axis'
            ext.header['BUNIT'] = '[AA]'
            fits.append(outfile, ext.data, ext.header)

            if metadata is not None:
                metext = fits.BinTableHDU(metadata)
                metext.header['EXTNAME'] = 'METADATA'
                fits.append(outfile, metext.data, metext.header)


        return outfile

    @staticmethod
    def plot_inversion_overview(filename: str) -> None:
        """
        Plots inversion results overview from a Level-2 HDUList

        Parameters
        ----------
        filename : str
            Path to a FITS file containing the inversion results.

        Returns
        -------
        None
        """
        params = {
            "savefig.dpi": 300,
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.titleweight": "bold",
            "axes.titlesize": 14,
            "font.size": 12,
            "legend.fontsize": 12,
            "font.family": "serif",
            "image.origin": "lower"
        }
        plt.rcParams.update(params)

        with fits.open(filename) as hdul:
            extnames = [hdul[i].header['EXTNAME'] for i in range(len(hdul))]
            chromolist = [i for i in extnames if "CHROMOSPHERE" in i]
            photolist = [i for i in extnames if "PHOTOSPHERE" in i]
            chisq = hdul['CHISQ'].data
            extlist = chromolist + photolist
            for extname in extlist:
                if "PHOTOSPHERE" in extname:
                    tau_zero = list(hdul[extname].data['logTau']).index(np.abs(hdul[extname].data['logTau']).min())
                else:
                    tau_zero = 0
                fig = plt.figure(figsize=(15, 10))
                gs = fig.add_gridspec(ncols=4, nrows=2, hspace=0.2, wspace=0.4)
                ax_bz = fig.add_subplot(gs[0, 0])
                ax_bx = fig.add_subplot(gs[0, 1])
                ax_by = fig.add_subplot(gs[0, 2])
                ax_v = fig.add_subplot(gs[1, 0])
                ax_t = fig.add_subplot(gs[1, 1])
                ax_chi = fig.add_subplot(gs[1, 2])

                plot_exts = [0, hdul[0].header['FOVY'], 0, hdul[0].header['FOVX']]

                # Bz, Bx, By

                bz = hdul[extname].data['Bz'][tau_zero]
                bzmap = ax_bz.imshow(
                    bz,
                    cmap='PuOr',
                    extents=plot_exts,
                    vmin=-3*np.nanstd(np.abs(bz)),
                    vmax=3*np.nanstd(np.abs(bz))
                )
                plt.colorbar(mappable=bzmap, ax=ax_bz)
                ax_bz.set_title("Bz [Gauss]")
                ax_bz.set_ylabel("Extent [arcsec]")
                ax_bz.set_xticks([])

                bx = hdul[extname].data['Bx'][tau_zero]
                bxmap = ax_bx.imshow(
                    bx,
                    cmap='PuOr',
                    extents=plot_exts,
                    vmin=-3 * np.nanstd(np.abs(bx)),
                    vmax=3 * np.nanstd(np.abs(bx))
                )
                plt.colorbar(mappable=bxmap, ax=ax_bx)
                ax_bx.set_title("Bx [Gauss]")
                ax_bx.set_yticks([])
                ax_bx.set_xticks([])

                by = hdul[extname].data['By'][tau_zero]
                bymap = ax_by.imshow(
                    by,
                    cmap='PuOr',
                    extents=plot_exts,
                    vmin=-3 * np.nanstd(np.abs(by)),
                    vmax=3 * np.nanstd(np.abs(by))
                )
                plt.colorbar(mappable=bymap, ax=ax_by)
                ax_by.set_title("By [Gauss]")
                ax_by.set_yticks([])
                ax_by.set_xticks([])

                # Velocity, beta, chi2
                v = hdul[extname].data['v'][tau_zero]
                vmap = ax_v.imshow(
                    v,
                    cmap='seismic',
                    extents=plot_exts,
                    vmin=-3*np.nanstd(np.abs(v)),
                    vmax=3*np.nanstd(np.abs(v))
                )
                plt.colorbar(mappable=vmap, ax=ax_v)
                ax_v.set_title("v [km/s]")
                ax_v.set_xlabel("Extent [arcsec]")
                ax_v.set_ylabel("Extent [arcsec]")
                if "PHOTOSPHERE" in extname:
                    key = 'T'
                    name = "T [K]"
                else:
                    key = "beta"
                    name = "Plasma-$\\beta$"
                t = hdul[extname].data[key][tau_zero]
                tmap = ax_t.imshow(
                    t,
                    cmap='hot',
                    extents=plot_exts,
                    vmin=np.nanmean(t) - np.nanstd(t),
                    vmax=np.nanmean(t) + np.nanstd(t)
                )
                plt.colorbar(mappable=tmap, ax=ax_t)
                ax_t.set_title(name)
                ax_t.set_yticks([])
                ax_t.set_xlabel("Extent [arcsec]")

                chimap = ax_chi.imshow(
                    chisq,
                    cmap='viridis',
                    extent=plot_exts,
                    vmin=0,
                    vmax=5*np.nanmedian(chisq)
                )
                plt.colorbar(mappable=chimap, ax=ax_chi)
                ax_chi.set_title("$\\chi^2$")
                ax_chi.set_yticks([])
                ax_chi.set_xlabel("Extent [arcsec]")

                fig.suptitle("Inversion Summary for {0}".format(extname))
                filestem = os.path.splitext(filename)[0]
                filename = filestem + "_{0}_inversion_summary.png".format(extname)
                plt.savefig(filename, bbox_inches='tight')
                plt.clf()
        return

    @staticmethod
    def inversion_picker(input_profile, wavelength_grid):
        """
        Pops up a series of matplotlib widget plots to choose inversion parameters from directly.
        Since we can pull any relevant invertable spectral lines from elsewhere, we won't bother
        making the user define them. Since telluric lines aren't stored in any of our inversion
        codes, however, we'll have the user grab those.

        Parameters
        ----------
        input_profile : numpy.ndarray
            Averaged spectral profile to choose ROI from
        wavelength_grid : numpy.ndarray
            Corresponding wavelength grid

        Returns
        -------
        spectral_range : numpy.ndarray
            Containing lambda_min, lambda_max
        continuum_range : numpy.ndarray
            Containing lambda_min, lambda_max (does not need to be contiguous to spectral_range)
        telluric_centers : numpy.ndarray
            Contatining the floating-point line center of each telluric line to throw into the mix
        """

        print("Select a minimum and maximum of the analysis region, then of a continuum region.")
        ranges = spex.select_spans_singlepanel(
            input_profile, xarr=wavelength_grid,
            fig_name="Select min/max for analysis, then of continuum",
            n_selections=2
        )
        spectral_range = ranges[0, :]
        continuum_range = ranges[1, :].reshape(1, 2) # Default ranges have 2 dimensions
        wvl_range = wavelength_grid[int(spectral_range.min()): int(spectral_range.max())]
        spex_range = input_profile[int(spectral_range.min()): int(spectral_range.max())]
        print("Select telluric line centers. Close when done.")
        telluric_centers = spex.select_lines_singlepanel_unbound_xarr(
            spex_range, xarr=wvl_range,
            fig_name="Select Telluric Lines"
        )
        telluric_center_indices = [spex.find_nearest(wvl_range, i) for i in telluric_centers]
        telluric_centers = np.array(
            [spex.find_line_core(
                spex_range[int(i-5):int(i+7)],
                wvl=wvl_range[int(i-5):int(i+7)]
            ) for i in telluric_center_indices]
        )

        return spectral_range, continuum_range, telluric_centers

    @staticmethod
    def get_reference_profile(fits_file: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Opens a fits file, creates a reference profile, grabs a wavelength grid.

        Parameters
        ----------
        fits_file : str
            Path to a compatible FITS file

        Returns
        -------
        reference_profile : numpy.ndarray
        wavelength_grid : numpy.ndarray
        """
        with fits.open(fits_file) as hdul:
            reference_profile = hdul['STOKES-I'].data.mean(axis=(0, 1))
            wavelength_grid = hdul['lambda-coordinate'].data
        return reference_profile, wavelength_grid

    @staticmethod
    def assemble_coord_grid(
            x_center: float, y_center: float, x_fov: float, y_fov: float,
            dx: float, dy: float, rotation: float, data_shape: tuple
    ) -> np.ndarray:
        """
        Assembles grid of theta/phi/gamma coordinates for a given telescope pointing.
        Parameters
        ----------
        x_center : float
        y_center : float
        x_fov : float
        y_fov : float
        dx : float
        dy : float
        rotation : float
        data_shape : tuple

        Returns
        -------
        coordinate_grid : numpy.ndarray

        """

        xy_grid = np.zeros((2, *data_shape))
        beta = np.arctan(y_fov / x_fov) * (180. / np.pi)
        delta = (360. - rotation) - beta
        half_diag = 0.5*np.sqrt(x_fov ** 2 + y_fov ** 2)
        diag_dy = half_diag * np.sin(np.pi * delta/180)
        diag_dx = half_diag * np.cos(np.pi * delta/180)
        xy_grid[:, 0, 0] = (x_center - diag_dx, y_center - diag_dy)
        for y in range(xy_grid.shape[1]):
            row_x0 = xy_grid[0, 0, 0] - y * dy * np.cos(np.pi/180 * (90 - rotation))
            row_y0 = xy_grid[1, 0, 0] + y * dy * np.sin(np.pi/180 * (90 - rotation))
            xy_grid[0, y, 0] = row_x0
            xy_grid[1, y, 0] = row_y0
            for x in range(xy_grid.shape[2]):
                col_x = xy_grid[0, y, 0] + x * dx * np.sin(np.pi/180 * (90 - rotation))
                col_y = xy_grid[1, y, 0] + x * dx * np.cos(np.pi/180 * (90 - rotation))
                xy_grid[0, y, x] = col_x
                xy_grid[1, y, x] = col_y
        alpha = 180 * np.arctan(xy_grid[0, :, :] / xy_grid[1, :, :]) / np.pi
        theta = np.arcsin(
            np.sqrt(
                xy_grid[0, :, :]**2 + xy_grid[1, :, :]**2
            ) / 960. # Fudge -- solar radius in arcsec
        ) * 180/np.pi
        gamma = 360 - (90 + alpha)
        for x in range(gamma.shape[0]):
            for y in range(gamma.shape[1]):
                gamma[x, y] = 90 - alpha[x, y] if (gamma[x, y] < 0) or (gamma[x, y] > 180) else gamma[x, y]
        phi = 0

        coordinate_grid = np.zeros((3, *gamma.shape))
        coordinate_grid[0] = theta
        coordinate_grid[1] = phi
        coordinate_grid[2] = gamma

        return coordinate_grid

    def center_to_limb_variation(self, theta: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
        """
        Builds a grid for center-to-limb intensity variation based on the projection angle, mu.

        Parameters
        ----------
        theta : numpy.ndarray
            grid of theta projection values (from assembleCoordGrid)
        wavelength : numpy.ndarray
            Wavelength grid in angstrom

        Returns
        -------
        clv_grid : numpy.ndarray
        """
        mu = np.cos(theta * np.pi/180)
        clv_grid = self.i0_allen(wavelength, mu) / self.i0_allen(wavelength, 1.0)

        return clv_grid

    @staticmethod
    def i0_allen(wavelength: float or np.ndarray, mu_angle: float or numpy.ndarray) -> float or np.ndarray:
        """
        Adaptation of the Allen model included with Hazel.
        I took the hardcoded data arrays, packed them into .npy save files, and packed it here.

        Parameters
        ----------
        wavelength : float or numpy.ndarray
            In Angstrom, wavelength to get CLV variation at
        mu_angle : float ot numpy.ndarray
            The cosine of the heliocentric angle (i.e., position on disk)

        Returns
        -------
        float or numpy.ndarray
            Interpolated CLV factor
        """
        with resources.path("ssosoft.spectral.inversions", "i0.npy") as rpath:
            i0 = np.load(rpath)
        with resources.path('ssosoft.spectral.inversions', 'lambdaI0.npy') as rpath:
            lambda_i0 = np.load(rpath)
        with resources.path('ssosoft.spectral.inversions', 'lambdaIC.npy') as rpath:
            lambda_ic = np.load(rpath)
        with resources.path('ssosoft.spectral.inversions', 'uData.npy') as rpath:
            u_data = np.load(rpath)
        with resources.path('ssosoft.spectral.inversions', 'vData.npy') as rpath:
            v_data = np.load(rpath)
        u = np.interp(wavelength, lambda_ic, u_data)
        v = np.interp(wavelength, lambda_ic, v_data)
        i0_interp = np.interp(wavelength, lambda_i0, i0)

        if type(mu_angle) is float:
            return (1.0 - u - v + u*mu_angle + v*mu_angle**2) * i0_interp
        else:
            mu_grid = np.repeat(mu_angle[:, :, np.newaxis], wavelength.shape[0], axis=-1)
            return (1.0 - u - v + u*mu_grid + v*mu_grid**2) * i0_interp


    @staticmethod
    def _bin_array(data: np.ndarray, axis: int, bin_value: int, bin_func: callable) -> np.ndarray:
        """
        Bins data along axis by some value

        Parameters
        ----------
        data : numpy.ndarray
            Binnable data
        axis : int
            Axis to bin data over
        bin_value : int
            Bin value
        bin_func : callable
            The function to use for binning. numpy.sum, numpy.mean, etc...

        Returns
        -------
        data : numpy.ndarray
            Binned data
        """
        dims = np.array(data.shape)
        argdims = np.arange(data.ndim)
        argdims[0], argdims[axis] = argdims[axis], argdims[0]
        data = data.transpose(argdims)
        data = [
            bin_func(
                np.take(
                    data,
                    np.arange(int(b * bin_value), int(b * bin_value + bin_value)),
                    0
                ),
                0
            )
            for b in np.arange(dims[axis] // bin_value)
        ]
        data = np.array(data).transpose(argdims)
        return data