# hsgPy for SSOSoft
The Horizontal Steerable Grating system at the Dunn Solar Telescope 
is a configurable spectrograph. The instrument is frequently reconfigured
in order to cover different spectral windows at variable cadences.
This package serves as a general calibration pipeline for the instrument, 
and is managed through a configuration file. 

The only assumption made about the system is the presence of certain canonical header
keywords in the Level-0 data product, and that the DST Camera Control systems will place
Level-0 slit images into FITS files with no data in the 0th header, and subsequent headers
containing data of the shape (1, ny, nx).

Performing calibrations is done by setting up a configuration file, then in python:
```python
import ssosoft.spectral.hsgCal as hsg
h = hsg.hsgCal("CAMERA_NAME", "configfile.ini")
h.hsg_run_calibration()
```
## hsgPy Reduction Steps:
The current iteration of hsgPy is (unfortunately) heavily reliant on widgets, due to the configurable
nature of the instrument, and the lack of day-to-day consistency in camera setups and precise spectral line locations.
A user running hsgPy will encounter up to three instances where the code requires user input: 
- Once to select two lines each from the flat image and the FTS atlas reference profile. 
  - The first line selected from the flat is used to create the gain table.
  - The others are used for wavelength calibration
- Again if Fourier fringe correction is requested to set the frequency cutoff
- Lastly to select spectral lines for velocity maps that will be packaged in the reduced file.
From these user inputs and the supplied configuration file, hsgPy will:
- Formulate and save average darks, solar flats, and lamp flats (if available)
- Attempt to determine beam edges and hairline positions
- Create the solar gain table
- Save calibration images (dark/solar flat/lamp flat/gain/skew along slit)
- Perform a wavelength calibration from the solar flat against the FTS atlas
- Perform a prefilter/grating efficiency correction (exclusive with fringe correction)
- Create a fringe template from the solar flat
  - Note that the gain table creation tends to wash out static fringes. Using the flat field to correct these should be valid.
  - This also works for FIRS. This will not work for SPINOR.
- Reduce each raster by, per slit position:
  - Performing dark, lamp, gain, prefilter (if available) calibration
  - Deskewing the slit image using the flat field skews
  - Performing an additional bulk shift along the wavelength axis to align the slit image with the flat image.
    - This ensures a valid wavelength calibration and that the fringe template is rigidly aligned.
  - Applies the fringe template 
  - Formulates velocity maps for given spectral lines
  - Packages raster, velocity maps, and other metadata information (wavelength, pointing, etc) in FITS format.

## hsgPy Level-1 Data Product
hsgPy packages reduced data in FITS format with comprehensive header information. 
The structure of the output file is (indexing from 0):
- Extension 0: Reduced data cube with (python) shape (ny, nx, nlambda)
- Extension 1 -- -1: Velocity maps for each selected spectral line
- Final extension: Metadata extension. In FITS table format, the following key/array pairs are stored:
  - "WAVELNGTH": Wavelength array of size nlambda for use with extension 0
  - "T_ELAPSED": Time since midnight in seconds of each slit position exposure start time
  - "EXPTIMES": Exposure time per slit position in ms. This is typically the same value
  - "STONYLAT": Stonyhurst Latitude of the center of each raster at each timestamp. The Sun is rotating.
  - "STONYLNG": Longitude of the same
  - "CROTAN": Rotation relative to Solar-North
  - "LIGHTLVL": The (unitless) amount of light seen by the DST guider. This can help the user filter out clouds, and correct continuum levels across slit positions, which is NOT a currently-implemented calibration step.
  - "SCINT": Values from the DST Seykora scintillation monitor at each slit position in arcsec. Note that typically AO is operated in conjunction with HSG, so these values should not be taken as the effective resolution of the telescope, but rather a quick reference.

## Configuration File for hsgPy
hsgPy works from a configuration file. There is a sample file included in this repository.
Headings are of the form [CAMERA_NAME]. Any name will work. Each camera has 17 keywords 
for the reduction process. These are:
- baseDir: Path to level-0 files
- reduceDir: path for calibration/reduced files
- reducedFilePattern: Naming convention for reduced files. Code expects there to be three {} tags for date, time, file number
- solarFlatPattern: naming convention for Solar flats. Glob notation, so \*solar_flat\* should work
- lampFlatPattern: Naming convention for lamp flats (may not exist on every day)
- darkPattern
- dataPattern
- reverseWave: "True" or "False". True reverses the spectral images along the x (lambda) axis.
- centralWavelength: The central wavelength (approximate, in angstrom) of the observing seriers
- approximateDispersion: Angstroms/pixel approximate
- arcsecPerPixelY: Until I can get the Hough transforms working on line grid images, this is required
- slitwidth: in microns
- beamEdgeThreshold: as a fraction of the flat field median value. 0.6 is usually sufficient.
- lineSelection: Currently, only "manual" is accepted (see below)
- prefilterDegree: Degree of polynomial to use for prefilter correction. Set to 0 for no correction. Do not use in conjunction with fringe correction
- fringeCorrection: "Fourier" or "None". Fourier does a simple fourier frequency filtering on the flat field to create a fringe template.
- fringeWavelength (optional): User can provide the wavelength for the Fringe determination.

## spectraTools.py
This file forms a set of common tools for spectrograph calibrations. These should work for any DST spectrograph
(or other solar spectrograph more generally), and will be reused for the upcoming firsPy reduction pipeline.
The included functions are:
- find_nearest & rolling_median: Helper functions
- fts_window: General function to pull a slice of the FTS atlas (indluded in the directory of that name).
- find_line_core: Fourier phase method to find a spectral line's core from a narrow slice in wavelength space.
- select_lines_singlepanel, select_lines_singlepanel_unbound_xarr, & select_lines_doublepanel: Widgets for user spectral line selection
- select_fringe_freq: Widget for user selection of frequency for Fourier fringe detrend.
- spectral_skew: Takes a slice along the slit of a single spectral line, and determines the skew along the slit.
- detect_beams_hairlines: Detects the positions of beams (single- or multi-) as well as hairlines per beam.
  - Uses intensity thresholding, and could use a rewrite
- create_gaintables: From a solar flat field and index of a strong spectral line, creates a gain image.
  - Note that solar gain tables have a known issue where static fringes are not detrended due to the way the gain tables are formed. Lamp flats help.
- iterate_shifts & fit_profile: helper functions to create the gain tables
- prefilter_correction: Performs a prefilter/grating efficiency correction via polynomial fitting. 
- fourier_fringe_correction: Creates a fringe template