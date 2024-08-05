# SSOsoft

SSOsoft is a set of tools for reducing data from the Sunspot Solar Observatory's Dunn Solar Telescope.

**Please report any bugs to sellers@nmsu.edu**

## Features

SSOsoft currently supports the ROSA and HARDcam (aka Zyla) fast imaging systems, as well as the HSG spectrograph.

## ROSA/HARDcam
ROSA and HARDcam are handled by the
```python
ssosoft.imagers.rosaZylaCal
ssosoft.imagers.kisipWrapper
ssosoft.imagers.destretch
```
submodules.

### Instrument Description
The Rapid Oscillations in the Solar Atmosphere (ROSA) instrument is a centrally-triggered fast imaging system.
The system can support up to 6 cameras, each with 1002x1004 pixel, 8x8 mm chips. The system is capable of up to 30 fps,
with various cameras dividing that master cadence to increase the exposure time while remaining synced. 
The H-Alpha Rapid Dynamics camera (HARDcam) is typically operated in conjunction with ROSA, using the same initial 
trigger pulse, in order to have a sync point at the start of a given observing series. 
HARDcam uses a newer Andor Zyla camera, with a 2048x2048 pixel, 13.3x13.3 mm chip.
HARDcam and Zyla are used interchangeably in documentation, but the system should be referred to as HARDcam in publications.

Typical filters for ROSA include the G-band 430 nm broadband filter, the 417 nm broadband filter, 
and the 0.12 nm FWHM Ca II K filter. If a fourth filter is used, it is typically either a 350 nm broadband filter, 
or a 0.025 nm FWHM slice pulled off of the facility's Universal Birefringeant Filter (UBF).

The HARDcam system uses the 0.025 nm FWHM Zeiss H-alpha Lyot-type filter.

### SSOsoft for ROSA/Zyla


ROSA and HARDcam are currently supported through to a level-1.5 data product. 
This package incorporates handling for flat/dark corrections, and a wrapper for the 
Kipenheuer-Institut Speckle Interferometry Package (KISIP, F. Woeger, National Solar Observatory).

SSOsoft will write image bursts for KISIP and trigger the program to run, provided the user has it locally available.
Once KISIP has finished with reconstructions, SSOsoft will re-convert the images to FITS standard, and apply destretch
techniques for image registration. A flow-preserving add-on is available, but not extensively tested.

The calibration pipeline requires a configuration file (see rosazyla_sampleConfig.ini in the repository base directory).
The configuration file may contain several different camera sections, plus sections for shared parameters.

With the configuration file set up, with a camera section named "CAMERA_NAME", in python:
```python
import ssosoft.imagers as img
r = img.rosaZylaCal.rosaZylaCal("CAMERA_NAME", "configFile.ini")
r.rosa_zyla_run_calibration()
k = img.kisipWrapper.kisipWrapper(r)
k.kisip_despeckle_all_batches()
d = img.destretch.rosaZylaDestretch(['CAMERA_NAME', 'CAMERA_NAME2'], "configFile.ini")
d.perform_destretch()
```
The "rosaZylaDestretch" class can take either a single camera, or a list of cameras. 
The destretch keywords in the configuration file can be set up to destretch one camera using the grid determined from a
reference camera. This is useful for, e.g., destretching chromospheric data relative to a photospheric channel in order
to preserve chromospheric flows. Currently, this can only be done for cameras with the same chip size 
(i.e., ROSA warps cannot be applied to HARDcam). Please make sure your plate scales are the same between cameras. 
Until I can get the Hough transform working on the line grid images for a blind plate scale determination, the software
is blind to the actual plate scale.

## HSG
HSG is handled by the 
```python
ssosoft.spectral.hsgCal
```
file, with common spectrograph tools contained in
```python
ssosoft.spectral.spectraTools
```
These common tools can be used by FIRS or SPINOR as well as HSG.

### Instrument Description

The Horizontal Steerable Grating (HSG) system at the Dunn Solar Telescope 
is a configurable spectrograph. The instrument is frequently reconfigured
in order to cover different spectral windows at variable cadences. 
Any number of cameras can be used with the system, but typically, the system is 
set up to use three spectral cameras, plus a slit-jaw imager, which is NOT handled by
this package.

### SSOsoft for HSG
This package serves as a general calibration pipeline for the instrument, 
and is managed through a configuration file. See hsg_sampleConfig.ini in the repository base directory.

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

The current iteration of hsgPy is (unfortunately) heavily reliant on widgets, due to the configurable
nature of the instrument, and the lack of day-to-day consistency in camera setups and precise spectral line locations.
A user running hsgPy will encounter up to three instances where the code requires user input: 
- Once to select two lines each from the flat image and the FTS atlas reference profile. 
  - The first line selected from the flat is used to create the gain table.
  - The others are used for wavelength calibration
- Again if Fourier fringe correction is requested to set the frequency cutoff
  - **Note:** My testing shows that the Fourier fringe correction RARELY works well. When in doubt, don't bother. 
  - I hope to fully replace the fringe correction with a better version in future releases. 
- Lastly to select spectral lines for velocity maps that will be packaged in the reduced file.
From these user inputs and the supplied configuration file, hsgPy will:
- Formulate and save average darks, solar flats, and lamp flats (if available)
- Attempt to determine beam edges and hairline positions
- Create the solar gain table
- Save calibration images (dark/solar flat/lamp flat/gain/skew along slit)
- Perform a wavelength calibration from the solar flat against the FTS atlas
- Perform a prefilter/grating efficiency correction (exclusive with fringe correction)
- Create a fringe template from the solar flat
- Reduce each raster by, per slit position:
  - Performing dark, lamp, gain, prefilter (if available) calibration
  - Deskewing the slit image using the flat field skews
  - Performing an additional bulk shift along the wavelength axis to align the slit image with the flat image.
    - This ensures a valid wavelength calibration and that the fringe template is rigidly aligned.
  - Applies the fringe template 
  - Formulates velocity maps for given spectral lines
  - Packages raster, velocity maps, and other metadata information (wavelength, pointing, etc) in FITS format.

The resulting Level-1.5 data product is a bit less user-friendly than the Level-1.5 ROSA/HARDcam data product, where 
the corrections are baked into the time series, and each reduced file is just a single image extension with a decent header.

### hsgPy Level-1.5 Data Product
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
