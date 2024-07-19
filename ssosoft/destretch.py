import configparser

from pyDestretch import *
import numpy as np
import cv2
import glob
import astropy.io.fits as fits
from tqdm import tqdm
import scipy.ndimage as scindi
import os
import dask.array as da
import dask


@dask.delayed
def _med_uni_filt(array1d, median_window, uniform_window, coord1, coord2, coord3):
    """Helper dask function that performs a median, then uniform filter on a 1d array of data.
    It then returns the 1d array, as well as its location in the original array, which are given as arguments.
    See the flow-removal functions below."""
    mf = scindi.median_filter(array1d, size=median_window, mode='nearest')
    uf = scindi.uniform_filter1d(mf, uniform_window, mode='nearest')
    ret_list = [uf, coord1, coord2, coord3]
    return ret_list


def _medfilt_wrapper(array, window):
    return scindi.median_filter(array, size=(1, 1, 1, window), mode='nearest')


def _unifilt_wrapper(array, window):
    return scindi.uniform_filter1d(array, window, mode='nearest', axis=-1)


class rosaZylaDestretch:
    """

    """
    def __init__(self, instruments, configFile, experimental="N"):
        """Unified SSOC Destretch module. Pass a configuration file (modified version of SSOSoft config file) and an
        instrument name, or list of instrument names. If it's a single instrument, it'll perform the iterative destretch
        defined by the config file. If it's a list of instruments, it'll perform the destretch specified in the config
        file for all instruments in the list. The reference channel, and other channels to be destretched to it are
        defined in the config file, as are bulk translations, etc.

        Args:
            instruments (list of str or str):
            configFile (str): Configuration file for initialization of SSODestretch
        """

        if type(instruments) == str:
            try:
                assert (instruments.upper() in ['ZYLA', 'ROSA_3500', 'ROSA_4170', 'ROSA_CAK', 'ROSA_GBAND']
                        ), ('Allowed values for <instrument>: '
                            'ZYLA, ROSA_3500, ROSA_4170, '
                            'ROSA_CAK', 'ROSA_GBAND'
                            )
            except Exception as err:
                print("Exception {0}".format(err))
                raise
        elif type(instruments) == list:
            for instrument in instruments:
                try:
                    assert (instrument.upper() in ['ZYLA', 'ROSA_3500', 'ROSA_4170', 'ROSA_CAK', 'ROSA_GBAND']
                            ), ('Allowed values for <instrument>: '
                                'ZYLA, ROSA_3500, ROSA_4170, '
                                'ROSA_CAK', 'ROSA_GBAND'
                                )
                except Exception as err:
                    print("Exception {0}".format(err))
                    raise
        try:
            f = open(configFile, mode='r')
            f.close()
        except Exception as err:
            print("Exception: {0}".format(err))
            raise

        self.configFile = configFile
        self.instruments = instruments
        self.referenceChannel = ""
        self.workBase = ""
        self.dstrFrom = ""
        self.hdrBase = ""
        self.speckleBase = ""
        self.speckleFileForm = ""
        self.postSpeckleBase = ""
        self.postDeflowBase = ""
        self.postDestretchBase = ""
        self.kernels = None
        self.dstrBase = ""
        self.dstrFlows = ""
        self.dstrVectorList = []
        # Relative to reference channel
        self.bulkTranslation = 0  # flipud, fliplr, flip, 0, or other angle
        self.theta = 0
        self.offset = (0, 0)
        self.magnification = 1
        self.dstrMethod = ""
        self.dstrWindow = 0
        self.channel = ""
        self.dataShape = (0, 0)
        self.flowWindow = 0
        self.date = ""
        self.time = ""
        self.dstrFilePattern = ""
        self.hdrList = []
        self.burstNum = ""
        self.experimental = experimental
        self.repair_tolerance = 0
        self.wave = ""
        self.exptime = ''
        # New as of 2024-01-05, list of translations required to get images pointed north.
        self.master_translation = []
        # For compatibility with older datasets that weren't destretched.
        # Going straight from speckle, we perform the master/bulk/fine translations in the speckle to fits function.
        # Then it doesn't need to be performed in the destretch function.
        # BUT older datasets don't HAVE speckle files, and go straight from postspeckle fits to spline destretch.
        # We still want those translations, so we have to do them in the destretch function where they're read in.
        # Since it's the same function handling both new and old, we set a flag that we fip to true if the translations
        # have already been performed, and use it on the second translation opportunity.
        self.master_translation_done = False

    def perform_destretch(self):
        """Perform SSOC destretch. You want a different destretch? You want flow-destructive? Code it yourself.
        pyDestretch has examples on github."""
        if type(self.instruments) == list:
            for channel in self.instruments:
                self.channel = channel
                if "ZYLA" in channel:
                    # Hardcoded for now.
                    self.dataShape = (2048, 2048)
                    self.destretch_zyla()
                else:
                    self.dataShape = (1002, 1004)
                    self.destretch_rosa()
        else:
            if "ZYLA" in self.instruments:
                self.destretch_zyla()
            else:
                self.destretch_rosa()
        return

    def destretch_rosa(self):
        self.configure_destretch()
        if self.dstrFrom.lower() == "speckle":
            if self.referenceChannel != self.channel:
                self.align_derotate_channels()
            self.speckleToFits()
        if self.referenceChannel == self.channel:
            self.destretch_reference()
            if self.flowWindow:
                self.remove_flows()
                self.apply_flow_vectors()
        if self.referenceChannel != self.channel:
            self.destretch_to_reference()
        return

    def destretch_zyla(self):
        self.configure_destretch()
        if self.dstrFrom.lower() == "speckle":
            self.speckleToFits()
        self.destretch_reference()
        if self.flowWindow:
            self.remove_flows()
            self.apply_flow_vectors()
        return

    def configure_destretch(self):
        """Set up destretch with relative channels, kernels, alignment params, rotation, etc.
        In here we need:
            Reference channel (if any)
            Header file base
            Final File Base
            Kernels
            If not Zyla, reference images for alignment/derotation
        """
        config = configparser.ConfigParser()
        config.read(self.configFile)
        self.dstrVectorList = []
        self.workBase = config[self.channel]['workBase']
        self.dstrFrom = config[self.channel]['dstrFrom']
        self.hdrBase = os.path.join(self.workBase, 'hdrs')
        self.hdrList = sorted(glob.glob(self.hdrBase + "/*.txt"))
        self.speckleBase = os.path.join(self.workBase, 'speckle')
        self.speckleFileForm = config[self.channel]['speckledFileForm']
        self.postSpeckleBase = os.path.join(self.workBase, "postSpeckle")
        self.postDestretchBase = os.path.join(self.workBase, "splineDestretch")
        c_dirs = [self.postSpeckleBase, self.postDestretchBase]
        for i in c_dirs:
            if not os.path.isdir(i):
                print("{0}: os.mkdir: attempting to create directory:""{1}".format(__name__, i))
                try:
                    os.mkdir(i)
                except Exception as err:
                    print("An exception was raised: {0}".format(err))
                    raise
        self.kernels = [int(i) for i in config[self.channel]['dstrKernel'].split(',')]
        self.referenceChannel = config[self.channel]['dstrChannel']
        self.flowWindow = config[self.channel]['flowWindow']
        if self.flowWindow.upper() != "NONE":
            self.flowWindow = int(self.flowWindow)
        else:
            self.flowWindow = None
        self.postDeflowBase = os.path.join(self.workBase, "flowPreservedDestretch")
        if (not os.path.isdir(self.postDeflowBase)) and (self.flowWindow is not None):
            print("{0}: os.mkdir: attempting to create directory:""{1}".format(__name__, self.postDeflowBase))
            try:
                os.mkdir(self.postDeflowBase)
            except Exception as err:
                print("An exception was raised: {0}".format(err))
                raise
        self.date = config[self.channel]['obsDate']
        self.time = config[self.channel]['obsTime']
        self.wave = config[self.channel]['wavelengthnm']
        self.exptime = config[self.channel]['expTimems']
        self.dstrFilePattern = config[self.channel]['destretchedFileForm']
        self.burstNum = config[self.channel]['burstNumber']
        # Keyword bulk translation can be a number, but may be a string.
        # Rather than write a function that attempts to identify the nature of the string,
        # We'll just use a fallback. Try to convert to float. If it fails, leave it as a string.
        # Also, you should be able to bulk translate the channel if it's the reference channel, so jot that down
        if config[self.channel]['bulkTranslation'].replace(".", "").replace("-", "").isnumeric():
            self.bulkTranslation = float(config[self.channel]['bulkTranslation'])
        else:
            self.bulkTranslation = config[self.channel]['bulkTranslation']
        if self.referenceChannel == self.channel:
            self.dstrBase = os.path.join(self.workBase, "destretch_vectors")
            c_dirs = [self.dstrBase]
            if self.flowWindow:
                self.dstrFlows = os.path.join(self.workBase, "destretch_vectors_noflow")
                c_dirs.append(self.dstrFlows)
            else:
                self.dstrFlows = None
            self.dstrMethod = config[self.channel]['dstrMethod']
            self.dstrWindow = int(config[self.channel]['dstrWindow'])
            for i in c_dirs:
                if not os.path.isdir(i):
                    print("{0}: os.mkdir: attempting to create directory:""{1}".format(__name__, i))
                    try:
                        os.mkdir(i)
                    except Exception as err:
                        print("An exception was raised: {0}".format(err))
                        raise
        else:
            self.dstrBase = os.path.join(config[self.referenceChannel]['workBase'], "destretch_vectors")
            if self.flowWindow:
                self.dstrFlows = os.path.join(config[self.referenceChannel]['workBase'], "destretch_vectors_noflow")
            if not os.path.isdir(self.dstrBase):
                print(
                    "Destretch: {0}, No destretch vectors found for reference channel: {1}".format(
                        self.channel,
                        self.referenceChannel
                    )
                )
                raise
        if "SHARED" in list(config.keys()):
            if "ROSA" in self.channel:
                if 'rosaTranslation'.lower() in list(config['SHARED'].keys()):
                    self.master_translation = config['SHARED']['rosaTranslation'].split(',')
            if "ZYLA" in self.channel:
                if 'zylaTranslation'.lower() in list(config['SHARED'].keys()):
                    self.master_translation = config['SHARED']['zylaTranslation'].split(',')

        # Lets the user define a more lenient tolerance for control point repair in destretch
        # Fully optional keyword, defaults to 0.3
        # If you're seeing a lot of flows "snapping" in your final product, maybe increase this value
        if 'repairTolerance' in [configPair[0] for configPair in config.items(self.channel)]:
            if config[self.channel]['repairTolerance'] == '':
                if "ZYLA" in self.channel:
                    self.repair_tolerance = 0.4
                elif "ROSA" in self.channel:
                    self.repair_tolerance = 0.3
                else:
                    self.repair_tolerance = 0.3
            else:
                self.repair_tolerance = float(config[self.channel]['repairTolerance'])
        elif "ZYLA" in self.channel:
            self.repair_tolerance = 0.4
        elif "ROSA" in self.channel:
            self.repair_tolerance = 0.3
        return

    def speckleToFits(self):
        """Function to dump speckle *.final files to fits format in the postSpeckle directory"""
        spklFlist = sorted(
            glob.glob(
                os.path.join(
                    self.speckleBase,
                    '*.final'
                )
            )
        )
        try:
            self.assert_flist(spklFlist)
        except AssertionError as err:
            print("Error: speckleList: {0}".format(err))
            raise

        alphaFlist = sorted(
            glob.glob(
                os.path.join(
                    self.speckleBase,
                    '*.subalpha'
                )
            )
        )
        try:
            self.assert_flist(alphaFlist)
        except AssertionError as err:
            print("Error: subalphaList: {0}".format(err))
            raise

        for i in tqdm(range(len(spklFlist)), desc="Converting Speckle to FITS"):
            spklImage = self.read_speckle(spklFlist[i])
            spklImage = self.perform_bulk_translation(spklImage)
            spklImage = self.perform_fine_translation(spklImage)
            spklImage = self.perform_master_translation(spklImage)
            hdrFile = self.hdrList[i]
            alpha = np.fromfile(alphaFlist[i], dtype=np.float32)[0]
            fname = os.path.join(
                self.postSpeckleBase,
                os.path.split(spklFlist[i])[-1] + '.fits'
            )
            self.write_fits(fname, spklImage, hdrFile, alpha=alpha, prstep=3)
        self.master_translation_done = True
        return

    def perform_bulk_translation(self, image):
        """Performs the bulk translation specified in config file."""
        if self.bulkTranslation == 0:
            # Base case, return image unaltered
            return image
        elif self.bulkTranslation == 'flipud':
            return np.flipud(image)
        elif self.bulkTranslation == 'fliplr':
            return np.fliplr(image)
        elif self.bulkTranslation == 'flip':
            return np.flip(image)
        elif ((type(self.bulkTranslation) == float) or
              (type(self.bulkTranslation) == int)) and (self.bulkTranslation != 0):
            return scindi.rotate(image, self.bulkTranslation, reshape=False, cval=image[0, 0])


    def perform_master_translation(self, image):
        """
        Performs the master translations for the instrument specified in the 'SHARED' section of the config file.
        """
        for i in self.master_translation:
            if "rot90" in i.lower():
                image = np.rot90(image)
            if "fliplr" in i.lower():
                image = np.fliplr(image)
            if "flipud" in i.lower():
                image = np.flipud(image)
        return image


    def perform_fine_translation(self, image):
        """Performs fine detrotation and shifts between cameras."""
        if (self.theta == 0) & (self.offset == (0, 0)):
            return image
        elif (self.theta == 0) & (self.offset != (0, 0)):
            return scindi.shift(image, self.offset, order=1, cval=image[0, 0])
        elif (self.theta != 0) & (self.offset == (0, 0)):
            return scindi.rotate(image, self.theta, reshape=False, cval=image[0, 0])
        else:
            return scindi.rotate(
                scindi.shift(
                    image, self.offset, order=1, cval=image[0, 0]
                ), self.theta, reshape=False, cval=image[0, 0]
            )

    def align_derotate_channels(self):
        """Currently only used in ROSA destretch. Finds bulk translation/rotation between a channel and its reference.
        """
        config = configparser.ConfigParser()
        config.read(self.configFile)
        reference_channel_reflist = sorted(glob.glob(
            os.path.join(
                config[self.referenceChannel]['refBase'],
                config[self.referenceChannel]['refFilePattern']
            )
        ))
        try:
            self.assert_flist(reference_channel_reflist)
        except AssertionError as err:
            print("Error: Reference Channel Reference list: {0}".format(err))
            raise

        channel_reflist = sorted(glob.glob(
            os.path.join(
                config[self.channel]['refBase'],
                config[self.channel]['refFilePattern']
            )
        ))
        try:
            self.assert_flist(reference_channel_reflist)
        except AssertionError as err:
            print("Error: Reference Channel Reference list: {0}".format(err))
            raise

        reffile = fits.open(reference_channel_reflist[0])
        if len(reffile) == 1:
            refim = reffile[0].data
        else:
            refim = reffile[1].data
        channelfile = fits.open(channel_reflist[0])
        if len(channelfile) == 1:
            chanim = self.perform_bulk_translation(channelfile[0].data)
        else:
            chanim = self.perform_bulk_translation(channelfile[1].data)

        self.determine_relative_rotation(chanim, refim)
        return

    def read_speckle(self, fname):
        """Read Speckle-o-gram"""
        return np.fromfile(fname, dtype=np.float32).reshape((self.dataShape[0], self.dataShape[1]))

    def destretch_reference(self):
        """Used when there is no reference channel, or when destretching the reference channel.
        Performs initial coarse destretch. Loops over the first 1 or 2 entries in self.kernels.
        1 entry, if leading kernel != 0 (i.e., no fine align)
        2 entry, if leading kernel == 0 (i.e., perform fine align)
        Write files to the self.dstrBase directory, and append filename to self.dstrVectorList.
        """
        postSpklFlist = sorted(
            glob.glob(
                os.path.join(
                    self.postSpeckleBase,
                    '*.fits'
                )
            )
        )
        try:
            self.assert_flist(postSpklFlist)
        except AssertionError as err:
            print("Error: speckleList: {0}".format(err))
            raise

        test_image = fits.open(postSpklFlist[0])[-1].data
        self.dataShape = (test_image.shape[0], test_image.shape[1])

        if self.dstrMethod == "running":
            reference_cube = np.zeros((self.dstrWindow, self.dataShape[0], self.dataShape[1]))
        elif self.dstrMethod == "reference":
            reference = fits.open(postSpklFlist[self.dstrWindow])[-1].data

        for i in tqdm(range(len(postSpklFlist)), desc="Destretching " + self.channel):
            # if self.dstrMethod == 'running':
            file = fits.open(postSpklFlist[i])
            img = file[-1].data
            if not self.master_translation_done:
                img = self.perform_bulk_translation(img)
                img = self.perform_master_translation(img)
            if (self.dstrMethod == 'running') & (i == 0):
                reference_cube[0, :, :] = img
                reference = reference_cube[0, :, :]
            elif (self.dstrMethod == 'running') & (i == 1):
                reference = reference_cube[0, :, :]
            elif (self.dstrMethod == 'running') & (i < self.dstrWindow):
                reference = np.nanmean(reference_cube[:i, :, :], axis=0)
            elif (self.dstrMethod == 'running') & (i > self.dstrWindow):
                reference = np.nanmean(reference_cube, axis=0)

            d_obj = Destretch(
                img,
                reference,
                self.kernels,
                return_vectors=True,
                repair_tolerance=self.repair_tolerance
            )
            dstr_im, dstr_vecs = d_obj.perform_destretch()
            if type(dstr_vecs) != list:
                dstr_vecs = [dstr_vecs]

            if self.dstrMethod == 'running':
                reference_cube[int(i % self.dstrWindow), :, :] = dstr_im

            # Write FITS and vectors
            dvec_name = os.path.join(self.dstrBase, str(i).zfill(5)+".npz")
            self.dstrVectorList.append(dvec_name)
            np.savez(dvec_name, *dstr_vecs)

            fname = os.path.join(
                self.postDestretchBase,
                self.dstrFilePattern.format(
                    self.date,
                    self.time,
                    i
                )
            )
            self.write_fits(fname, dstr_im, file[-1].header, prstep=4)
        return

    def apply_flow_vectors(self):
        """Apply flow-removed dstr vectors, write fits files to new directory."""
        postSpklFlist = sorted(
            glob.glob(
                os.path.join(
                    self.postSpeckleBase,
                    '*.fits'
                )
            )
        )
        try:
            self.assert_flist(postSpklFlist)
        except AssertionError as err:
            print("Error: postSpklFlist: {0}".format(err))
            raise

        deflowFlist = sorted(
            glob.glob(
                os.path.join(
                    self.dstrFlows,
                    '*.npz'
                )
            )
        )
        try:
            self.assert_flist(deflowFlist)
        except AssertionError as err:
            print("Error: deflowFlist: {0}".format(err))
            raise

        for i in tqdm(range(len(postSpklFlist)), desc="Appling de-flowed destretch"):
            target_file = fits.open(postSpklFlist[i])
            target_image = target_file[-1].data
            dstr_vec = np.load(deflowFlist[i])
            vecs = [dstr_vec[k] for k in dstr_vec.files]
            d = Destretch(target_image, target_image, self.kernels, warp_vectors=vecs)
            dstrim = d.perform_destretch()
            fname = os.path.join(
                self.postDeflowBase,
                self.dstrFilePattern.format(
                    self.date,
                    self.time,
                    i
                )
            )
            self.write_fits(fname, dstrim, target_file[-1].header, prstep=5)
        return

    def destretch_to_reference(self):
        """Destretch to a reference list of vectors
        # Step 0: Get list of files from self.dstrBase,
        #   which should be set up in the config file from self.referenceChannel
        # Step 1: From the list of destretch targets and the list of vectors, see if there's a dimension mismatch
        # Step 2: If there is a mismatch, divide the vector list len by the target len.
        #   Step 2.5: Use this to determine the vector list index by having a second iterable, and passing this iterable
        #   to round()
        # Step 3: Apply destretch
        # Step 4: Write FITS.
        """
        postSpklFlist = sorted(
            glob.glob(
                os.path.join(
                    self.postSpeckleBase,
                    '*.fits'
                )
            )
        )
        try:
            self.assert_flist(postSpklFlist)
        except AssertionError as err:
            print("Error: speckleList: {0}".format(err))
            raise

        self.dstrVectorList = sorted(
            glob.glob(
                os.path.join(
                    self.dstrBase,
                    '*.npz'
                )
            )
        )
        try:
            self.assert_flist(self.dstrVectorList)
        except AssertionError as err:
            print("Error: Vector List: {0}".format(err))
            raise

        if self.dstrFlows:
            deflowFlist = sorted(
                glob.glob(
                    os.path.join(
                        self.dstrFlows,
                        '*.npz'
                    )
                )
            )
            try:
                self.assert_flist(deflowFlist)
            except AssertionError as err:
                print("Error: Vector List: {0}".format(err))
                raise

        if len(postSpklFlist) != len(self.dstrVectorList):
            dstr_vec_increment = len(postSpklFlist)/len(self.dstrVectorList)
        else:
            dstr_vec_increment = 1
        dstr_ctr = 0
        for i in tqdm(range(len(postSpklFlist)), desc="Appling Destretch Vectors..."):
            file = fits.open(postSpklFlist[i])
            img = file[-1].data
            if not self.master_translation_done:
                img = self.perform_bulk_translation(img)
                img = self.perform_fine_translation(img)
                img = self.perform_master_translation(img)
            # Since we're using savez, we need to unpack the arrays into a list for destretch
            vecs = np.load(self.dstrVectorList[int(round(dstr_ctr))])
            vecs = [vecs[k] for k in vecs.files]
            d = Destretch(
                img,
                img,
                self.kernels,
                warp_vectors=vecs
            )
            dstrim = d.perform_destretch()
            fname = os.path.join(
                self.postDestretchBase,
                self.dstrFilePattern.format(
                    self.date,
                    self.time,
                    i
                )
            )
            self.write_fits(fname, dstrim, file[-1].header, prstep=4)

            if self.flowWindow:
                if len(deflowFlist) > 0:
                    vecs = np.load(deflowFlist[i])
                    vecs = [vecs[k] for k in vecs.files]
                    d = Destretch(
                        img,
                        img,
                        self.kernels,
                        warp_vectors=vecs
                    )
                    dstrim = d.perform_destretch()
                    fname = os.path.join(
                        self.postDeflowBase,
                        self.dstrFilePattern.format(
                            self.date,
                            self.time,
                            i
                        )
                    )
                    self.write_fits(fname, dstrim, file[-1].header, prstep=5)

            dstr_ctr += dstr_vec_increment
        return

    def write_fits(self, fname, data, hdr, alpha=None, prstep=4):
        """Write destretched FITS files."""
        allowed_keywords = [
            'DATE', 'STARTOBS', 'ENDOBS',
            'EXPOSURE', 'HIERARCH',
            'CRVAL1', 'CRVAL2',
            'CTYPE1', 'CTYPE2',
            'CUNIT1', 'CUNIT2',
            'CDELT1', 'CDELT2',
            'CRPIX1', 'CRPIX2',
            'CROTA2',
            'SCINT', 'LLVL',
            'RSUN_REF'
        ]
        float_keywords = [
            'CRVAL1', 'CRVAL2',
            'CROTA2',
            'SCINT', 'LLVL',
        ]
        asec_comment_keywords = [
        'CDELT1', 'CDELT2',
        'CRPIX1', 'CRPIX2',
        'RSUN_REF'
        ]
        if type(hdr) is fits.header.Header:
            hdul = fits.HDUList(fits.PrimaryHDU(data, header=hdr))
        else:
            hdul = fits.HDUList(fits.PrimaryHDU(data))
        prstep_flags = ['PRSTEP1', 'PRSTEP2', 'PRSTEP3', 'PRSTEP4', 'PRSTEP5']
        prstep_values = [
            'DARK-SUBTRACTION,FLATFIELDING',
            'SPECKLE-DECONVOLUTION',
            'ALIGN TO SOLAR NORTH',
            'DESTRETCHING',
            'FLOW-PRESERVING-DESTRETCHING'
        ]
        prstep_comments = [
            'SSOsoft',
            'KISIP v0.6',
            'SSOsoft',
            'pyDestretch',
            'pyDestretch with flow preservation'
        ]

        if type(hdr) is str:
            header = open(hdr, 'r').readlines()
            for i in range(len(header)):
                slug = header[i].split("=")[0].strip()
                field = header[i].split("=")[-1].split("/")[0]
                field = field.replace("\n","").replace("\'","").strip()
                if field.isnumeric():
                    field = float(field)
                if any(substring in slug for substring in allowed_keywords):
                    if "STARTOBS" in slug:
                        hdul[0].header['STARTOBS'] = (field, "Date of start of observation")
                        hdul[0].header['DATE_OBS'] = (field, "Date of start of observation")
                        hdul[0].header['DATE-BEG'] = (field, "Date of start of observation")
                        hdul[0].header['DATE'] = (np.datetime64('now').astype(str), "Date of file creation")
                    if "ENDOBS" in slug:
                        hdul[0].header['ENDOBS'] = (field, "Date of end of observation")
                        hdul[0].header['DATE-END'] = (field, "Date of end of observation")
                    elif "RSUN" in slug:
                        hdul[0].header['RSUN_ARC'] = (round(float(field)/2, 3), "Radius of Sun in arcsec")
                    elif any(substring in slug for substring in float_keywords):
                        hdul[0].header[slug] = round(float(field), 3)
                    else:
                        if any(substring in slug for substring in asec_comment_keywords):
                            hdul[0].header[slug] = (round(float(field), 3), 'arcsec')
                        else:
                            hdul[0].header[slug] = field
        hdul[0].header['BUNIT'] = 'DN'
        hdul[0].header['NSUMEXP'] = (self.burstNum, "Frames used in speckle reconstruction")
        hdul[0].header['TEXPOSUR'] = (self.exptime, "ms, Single-frame exposure time")
        hdul[0].header['AUTHOR'] = 'sellers'
        hdul[0].header['TELESCOP'] = "DST"
        hdul[0].header['ORIGIN'] = 'SSOC'
        if "ROSA" in self.channel.upper():
            hdul[0].header['INSTRUME'] = "ROSA"
        if "ZYLA" in self.channel.upper():
            hdul[0].header['INSTRUME'] = "HARDCAM"
        hdul[0].header['WAVE'] = self.wave
        hdul[0].header['WAVEUNIT'] = "nm"

        if alpha:
            hdul[0].header['SPKLALPH'] = alpha
        for i in range(prstep):
            hdul[0].header[prstep_flags[i]] = (prstep_values[i], prstep_comments[i])
        hdul.writeto(fname, overwrite=True)
        return

    def remove_flows(self):
        """Function to remove lateral solar flows from destretch vector.
        It does this by loading the target control points from the saved destretch parameters,
        then doing a median filter in the time-direction,
        subtracting this median off of the target control points.
        This runs over the last kernel."""
        destretch_coord_list = self.dstrVectorList
        smooth_number = self.flowWindow

        index_in_file = -1

        template_coords = np.load(destretch_coord_list[0])
        tpl_coord_shape = template_coords[template_coords.files[index_in_file]].shape
        shifts_all = np.zeros(
            (
                len(destretch_coord_list),
                *tpl_coord_shape
            )
        )
        shifts_corr_sum = np.zeros(
            (
                len(destretch_coord_list),
                *tpl_coord_shape
            )
        )
        translations = np.zeros((len(destretch_coord_list), 2))
        for i in tqdm(range(len(destretch_coord_list)), desc="Loading Destretch Vectors"):
            dstr = np.load(destretch_coord_list[i])
            tcpl = dstr[dstr.files[index_in_file]]
            # Ref. Control Point index depends on whether there are bulk shifts.
            # If there are an odd number of arrays in the file, there are bulk shifts
            if len(dstr) % 2 == 0:
                rcpl = dstr[dstr.files[int(len(dstr)/2) - 1]]
            else:
                rcpl = dstr[dstr.files[int((len(dstr)/2))]]
                translations[i] = dstr[dstr.files[0]]
            shifts_all[i] = tcpl - rcpl
            if i == 0:
                shifts_corr_sum[i] = shifts_all[i]
            else:
                shifts_corr_sum[i] = shifts_corr_sum[i - 1] + shifts_all[i]

        # Algorithm is:
        #   1.) Get cumulative sum of all shifts in x/y
        #   2.) Take median filter of cumulative sum
        #   3.) Take uniform filter of median-filtered cumulative sum
        #   4.) Subtract this uniform filter off the cumulative sum
        #   5.) Add the reference control points back on to cumulative sum
        #   6.) Save as target control points.

        median_filtered = scindi.median_filter(
            shifts_corr_sum,
            size=(smooth_number, 1, 1, 1),
            mode='nearest'
        )
        flows = scindi.uniform_filter(
            median_filtered,
            size=(smooth_number, 1, 1, 1),
            mode='nearest'
        )
        flow_detr_shifts = shifts_corr_sum - flows

        for i in tqdm(range(len(destretch_coord_list)), desc="Saving Flow-Detrended Vectors"):
            original_file = np.load(destretch_coord_list[i])
            original_arrays = [original_file[k] for k in original_file.files]
            if len(original_arrays) % 2 == 0:
                rcpl = dstr[dstr.files[int(len(dstr)/2) - 1]]
            else:
                rcpl = dstr[dstr.files[int((len(dstr)/2))]]
            original_arrays = original_arrays[:-1]
            original_arrays.append(flow_detr_shifts[i] + rcpl)
            writeFile = os.path.join(self.dstrFlows, str(i).zfill(5))
            np.savez(writeFile + '.npz', *original_arrays)

        return

    def remove_flows_alternate(self):
        """Alternate implementation of flow removal that utilizes dask delayed to parallelize the calculation of flows.
        I believe this will be faster than the other implementation, which uses dask arrays, but am not certain.
        """
        destretch_coord_list = self.dstrVectorList
        smooth_number = self.flowWindow
        if self.dstrMethod == 'reference':
            median_number = 10
        else:
            median_number = self.dstrWindow

        index_in_file = -1

        template_coords = np.load(destretch_coord_list[0])
        # If the shape of this is nk, 2, ny, nx, truncate to the last along the 0th axis
        if len(template_coords.shape) > 3:
            template_coords = template_coords[index_in_file, :, :, :]
        grid_x, grid_y = np.meshgrid(
            np.arange(template_coords.shape[-1]),
            np.arange(template_coords.shape[-2])
        )
        # Given that a typical observing day comprises ~1400 files, we cannot hold everything in memory.
        # So we should only load in the number corresponding to the flowWindow
        shifts_all = np.zeros(
            (
                template_coords.shape[0],
                template_coords.shape[1],
                template_coords.shape[2],
                smooth_number * 4
            ),
            dtype=np.float32
        )
        shifts_corr_sum = np.zeros(
            (
                template_coords.shape[0],
                template_coords.shape[1],
                template_coords.shape[2],
                smooth_number * 4
            ),
            dtype=np.float32
        )

        translations = np.zeros((2, smooth_number * 4), dtype=np.float32)

        for i in tqdm(range(len(destretch_coord_list)), desc="Removing Flows"):
            master_dv = np.load(destretch_coord_list[i]).astype(np.float32)
            master_coords = master_dv[index_in_file]
            master_coords[master_coords == -1] = np.nan
            # Explanation time.
            # The algorithm here has 4 base cases:
            #   1.) 0th iteration,
            #   2.) iteration number < smooth_number/2,
            #   3.) iteration number > smooth_number/2 AND iteration_number < number of files - smooth_number/2
            #   4.) iteration number > number of files - smooth_number/2
            # Essentially, the goal is to:
            #   a.) keep the current frame in the center of our window
            #   b.) Do as few calculations as possible.
            # 1.) So for i == 0, we set up our arrays for the first smooth_number files in the list.
            #   Do the uniform/median filter, and take the 0th index of the calculated flows.
            #   NOT the middle index, cause nothing's happened yet, flow-wise.
            # 2.) This case, we essentially do not iterate, just save progressive indices until we're in the center
            #   of the array. No need to recalculate.
            # 3.) Now we increment forward one. Shift the arrays so the 0th index is overwritten, and the last can
            #   be filled. Fill the last, recalculate the flows, take the middle index to save
            # 4.) Now there's no more data to fill. No need to increment or recalculate, just allow the save index
            #   to progress to the end of the array.
            if i == 0:
                # Have to pre-fill arrays on 0th step
                for j in tqdm(range(smooth_number * 4), desc="Setting up arrays"):
                    dvs = np.load(destretch_coord_list[j]).astype(np.float32)
                    translations[:, j] = dvs[0, :, 0, 0]
                    coords = dvs[index_in_file]
                    coords[coords == -1] = np.nan

                    shifts_all[0, :, :, j] = coords[0, :, :] - grid_y
                    shifts_all[1, :, :, j] = coords[1, :, :] - grid_x
                    # Testing, but I don't think we need bulk shifts.
                    # These should already be removed if the leading kernel is 0. If it isn't, feck em. [for now]
                    if j == 0:
                        shifts_corr_sum[:, :, :, j] = shifts_all[:, :, :, j]
                    else:
                        shifts_corr_sum[:, :, :, j] = shifts_corr_sum[:, :, :, j - 1] + shifts_all[:, :, :, j]
                index = i
                # I am becoming convinced that this is not as efficient as it could be.
                # The dask arrays are statically chunked, but we're losing a significant amount of time to
                # Conversion and chunking.
                # What if we went with a dask delayed approach instead?
                # Write a delayed function that takes:
                #   a 1-D array of smooth vectors
                #   their x-coordinate in the array
                #   their y-coordinate in the array
                #   their z-coordinate in the array
                # And returns an array containing:
                #   the flow value
                #   its x, y, and z coordinate
                # Run the dask delayed on each image (i.e., it'll do this 1M -- 4M times.
                # Then we have a list of these arrays. Reconstruct the flow array from the arrays contained in the list
                # And subtract from the shifts_corr_sum array as normal
                # Better write this in a separate function until we can verify it's faster.

                flow_list = []
                for c0 in range(shifts_corr_sum.shape[0]):
                    for c1 in range(shifts_corr_sum.shape[1]):
                        for c2 in range(shifts_corr_sum.shape[2]):
                            flow_list.append(
                                _med_uni_filt(
                                    shifts_corr_sum[c0, c1, c2, :],
                                    median_number,
                                    smooth_number,
                                    c0,
                                    c1,
                                    c2
                                )
                            )
                flows_flat = dask.compute(flow_list)[0]
                flows = np.zeros(shifts_corr_sum.shape)
                for f in flows_flat:
                    flows[f[1], f[2], f[3], :] = f[0]

                flow_detr_shifts = shifts_corr_sum - flows
            elif i < int(smooth_number * 2):
                index = i
            elif i > len(destretch_coord_list) - int(smooth_number * 2):
                index = i % (smooth_number * 2)
            else:
                index = int(smooth_number * 2)
                # Increment the flow arrays forward one after previous step...
                translations[:, :-1] = translations[:, 1:]
                shifts_all[:, :, :, :-1] = shifts_all[:, :, :, 1:]
                shifts_corr_sum[:, :, :, :-1] = shifts_corr_sum[:, :, :, 1:]

                # Fill last index with the coordinates smooth_number/2 after current iteration
                add_to_end = np.load(destretch_coord_list[i + int(smooth_number / 2) - 1])
                coords = add_to_end[index_in_file]
                translations[:, -1] = add_to_end[0, :, 0, 0]
                shifts_all[0, :, :, -1] = coords[0, :, :] - grid_y
                shifts_all[1, :, :, -1] = coords[1, :, :] - grid_x
                shifts_corr_sum[:, :, :, -1] = shifts_corr_sum[:, :, :, -2] + shifts_all[:, :, :, -1]

                corr_sum = da.from_array(shifts_corr_sum)
                corr_sum = corr_sum.rechunk({0: 'auto', 1: 'auto', 2: 'auto', 3: -1})

                median_filtered = corr_sum.map_overlap(
                    _medfilt_wrapper,
                    depth=0,
                    window=median_number).compute()
                median_filtered = da.from_array(median_filtered)
                median_filtered = median_filtered.rechunk({0: 'auto', 1: 'auto', 2: 'auto', 3: -1})
                flows = median_filtered.map_overlap(_unifilt_wrapper, depth=0, window=smooth_number).compute()

                flow_detr_shifts = shifts_corr_sum - np.array(flows)

            save_array = master_dv
            save_array[index_in_file, 0, :, :] = flow_detr_shifts[0, :, :, index] + grid_y
            save_array[index_in_file, 1, :, :] = flow_detr_shifts[1, :, :, index] + grid_x
            writeFile = os.path.join(self.dstrFlows, str(i).zfill(5))
            np.save(writeFile + ".npz", save_array)
        return

    def assert_flist(self, flist):
        assert (len(flist) != 0), "List contains no matches"
        return

    def determine_relative_rotation(self, image, reference):
        """Determines the relative rotation between two images, using feature matching from opencv.
        Ideally, this should be done using the target image, as it provides structure that is non-symmetric.
        This is not the case for the pinhole or line grid images, and use of the targets should improve the accuracy of
        the feature matching. We use OpenCV's implementation of FLANN to detect features
        Matching is done via a simple ratio test per Lowe (2004).
        The 2D transformation matrix is constructed from estimating Affines.
        This provides scaling, rotation, and translation, but it is unclear whether the image convolution approach
        is more accurate for translation.

        Args:
            image (array-like): The rotated image
            reference (array-like): The reference image
                NOTE: The 4170 camera seems to be more rotated than gband for the DST. While 4170 may seem a
                better choice of reference, I would recommend the g-band be used instead.

        Returns:
            theta (float): The rotation angle (in degrees)
            scale (tuple): The relative scaling as (sx, sy)
            shifts (tuple): The relative shift between images (dx, dy)
        """
        # OpenCV requires 8-bit images for some reason.
        reference_8bit = cv2.normalize(reference, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        # SIFT for keypoints/descriptions
        sift = cv2.SIFT_create()
        keypoints_img, description_img = sift.detectAndCompute(image_8bit, None)
        keypoints_ref, description_ref = sift.detectAndCompute(reference_8bit, None)

        # Setting up FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(description_ref, description_img, k=2)

        pts1 = []
        pts2 = []

        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                pts2.append(keypoints_img[m.trainIdx].pt)
                pts1.append(keypoints_ref[m.queryIdx].pt)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        transform, mask = cv2.estimateAffine2D(pts1, pts2)
        shifts = (transform[0, 2], transform[1, 2])
        scale = (np.sign(transform[0, 0]) * np.sqrt(transform[0, 0]**2 + transform[0, 1]**2),
                 np.sign(transform[1, 1]) * np.sqrt(transform[1, 0]**2 + transform[1, 1]**2))
        theta = np.arctan2(-transform[0, 1], transform[0, 0]) * 180/np.pi
        self.theta = theta
        self.offset = shifts
        self.magnification = scale

        return
