import configparser

from pyDestretch import *
import numpy as np
import cv2
import glob
import astropy.io.fits as fits
from tqdm import tqdm
import scipy.ndimage as scindi
import os


class rosaZylaDestretch:
    """

    """
    def __init__(self, instruments, configFile):
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
        self.deflowDir = ""
        self.referenceChannel = ""
        self.workBase = ""
        self.hdrBase = ""
        self.speckleBase = ""
        self.speckleFileForm = ""
        self.postSpeckleBase = ""
        self.kernels = None
        self.dstrBase = ""
        self.dstrVectorList = []
        self.spklList = []
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
        self.workBase = config[self.channel]['workBase']
        self.hdrBase = os.path.join(self.workBase, 'hdrs')
        self.hdrList = sorted(glob.glob(self.hdrBase + "/*.txt"))
        self.speckleBase = os.path.join(self.workBase, 'speckle')
        self.speckleFileForm = config[self.channel]['speckledFileForm']
        self.postSpeckleBase = os.path.join(self.workBase, "postSpeckle")
        self.kernels = [int(i) for i in config[self.channel]['dstrKernel'].split(',')]
        self.referenceChannel = config[self.channel]['dstrChannel']
        self.flowWindow = config[self.channel]['flowWindow']
        self.date = config[self.channel]['obsDate']
        self.time = config[self.channel]['obsTime']
        self.dstrFilePattern = config[self.channel]['destretchedFileForm']
        self.burstNum = config[self.channel]['burstNumber']
        if self.referenceChannel == self.channel:
            self.dstrBase = os.path.join(self.workBase, "destretch_vectors")
            self.dstrMethod = config[self.channel]['dstrMethod']
            self.dstrWindow = int(config[self.channel]['dstrWindow'])
            if not os.path.isdir(self.dstrBase):
                print("{0}: os.mkdir: attempting to create directory:""{1}".format(__name__, self.dstrBase))
                try:
                    os.mkdir(self.dstrBase)
                except Exception as err:
                    print("An exception was raised: {0}".format(err))
                    raise
        else:
            # Keyword bulk translation can be a number, but may be a string.
            # Rather than write a function that attempts to identify the nature of the string,
            # We'll just use a fallback. Try to convert to float. If it fails, leave it as a string.
            if config[self.channel]['bulkTranslation'].replace(".", "").replace("-", "").isnumeric():
                self.bulkTranslation = float(config[self.channel]['bulkTranslation'])
            else:
                self.bulkTranslation = config[self.channel]['bulkTranslation']

            self.dstrBase = os.path.join(config[self.referenceChannel]['workBase'], "destretch_vectors")
            if not os.path.isdir(self.dstrBase):
                print(
                    "Destretch: {0}, No destretch vectors found for reference channel: {1}".format(
                        self.channel,
                        self.referenceChannel
                    )
                )
                raise

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

    def read_speckle(self, fname):
        """Read Speckle-o-gram"""
        return np.fromfile(fname, dtype=np.float32).reshape((self.dataShape[0], self.dataShape[1]))

    def destretch_rosa(self):
        self.configure_destretch()
        if self.referenceChannel == self.channel:
            self.destretch_reference_coarse()
            self.remove_flows()
            self.destretch_reference_fine()
        if self.referenceChannel != self.channel:
            self.align_derotate_channels()
            self.destretch_to_reference()

    def destretch_zyla(self):
        self.configure_destretch()
        self.destretch_reference_coarse()
        self.remove_flows()
        self.destretch_reference_fine()

    def destretch_reference_coarse(self):
        """Used when there is no reference channel, or when destretching the reference channel.
        Performs initial coarse destretch. Loops over the first 1 or 2 entries in self.kernels.
        1 entry, if leading kernel != 0 (i.e., no fine align)
        2 entry, if leading kernel == 0 (i.e., perform fine align)
        Write files to the self.dstrBase directory, and append filename to self.dstrVectorList.
        """
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

        if self.dstrMethod == "running":
            reference_cube = np.zeros((self.dstrWindow, self.dataShape[0], self.dataShape[1]))
        elif self.dstrMethod == "reference":
            reference = self.read_speckle(spklFlist[self.dstrWindow])

        if len(self.kernels) > 1:
            if self.kernels[0] == 0:
                kernels = self.kernels[:2]
            else:
                kernels = self.kernels[:1]
        else:
            kernels = self.kernels
        l = "Coarse Destretching " + self.channel
        for i in tqdm(range(len(spklFlist)), desc=l):
            # if self.dstrMethod == 'running':
            img = self.read_speckle(spklFlist[i])
            if (self.dstrMethod == 'running') & (i == 0):
                reference_cube[0, :, :] = img
                reference = reference_cube[0, :, :]
            elif (self.dstrMethod == 'running') & (i == 1):
                reference = reference_cube[0, :, :]
                img = self.read_speckle(spklFlist[i])
            elif (self.dstrMethod == 'running') & (i < self.dstrWindow):
                reference = np.nanmean(reference_cube[:i, :, :], axis=0)
                img = self.read_speckle(spklFlist[i])
            elif (self.dstrMethod == 'running') & (i > self.dstrWindow):
                reference = np.nanmean(reference_cube, axis=0)

            d_obj = Destretch(
                img,
                reference,
                kernels,
                return_vectors=True
            )
            dstr_im, dstr_vecs = d_obj.perform_destretch()
            if type(dstr_vecs) != list:
                dstr_vecs = [dstr_vecs]

            if self.dstrMethod == 'running':
                reference_cube[int(i % self.dstrWindow), :, :] = dstr_im

            dvec_name = os.path.join(self.dstrBase, str(i).zfill(5)+".npy")
            self.dstrVectorList.append(dvec_name)
            np.save(dvec_name, dstr_vecs)

    def destretch_reference_fine(self):
        """Secondary Fine Destretch. If there are no additional kernel sizes to destretch after the deflow kernels,
        just apply the deflowed vectors and save the fits file."""
        # STEP -1: Cut kernel list to only fine kernels
        # STEP 0: Loop over speckle images
        # STEP 1: Apply destretch. Return image.
        # STEP 2: If there are no fine kernels, write the image + vectors
        # STEP 3: If there ARE fine kernels, destretch to fine grid
        # STEP 4: Write fine-kernel destretch vectors (just the fine kernels)
        # STEP 5: Write FITS

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

        if (self.kernels[0] == 0) & (len(self.kernels) > 2):
            kernels = self.kernels[2:]
            finish_dstr = False
        elif (self.kernels[0] != 0) & (len(self.kernels) > 1):
            kernels = self.kernels[1:]
            finish_dstr = False
        else:
            kernels = self.kernels
            finish_dstr = True

        coarse_dstr = []
        for i in tqdm(range(len(self.spklList)), desc="Appling de-flowed destretch"):
            target_image = self.read_speckle(self.spklList[i])
            dstr_vec = self.read_speckle(self.dstrVectorList[i])
            d = Destretch(target_image, target_image, self.kernels, warp_vectors=dstr_vec)
            dstrim = d.perform_destretch()
            fname = os.path.join(
                self.postSpeckleBase,
                self.dstrFilePattern.format(
                    self.date,
                    self.time,
                    i
                )
            )
            alpha = np.fromfile(alphaFlist[i], dtype=np.float32)[0]
            self.write_fits(fname, dstrim, self.hdrList[i], alpha=alpha)
            coarse_dstr.append(fname)

        if finish_dstr:
            return

        if self.dstrMethod == "running":
            reference_cube = np.zeros((self.dstrWindow, self.dataShape[0], self.dataShape[1]))
        elif self.dstrMethod == "reference":
            reference = fits.open(coarse_dstr[self.dstrWindow])[0].data

        for i in tqdm(range(len(coarse_dstr)), desc='Applying Fine Destretch'):
            vecs_master = np.load(self.dstrVectorList[i])
            img = fits.open(coarse_dstr[i])[0].data
            if (self.dstrMethod == 'running') & (i == 0):
                reference_cube[0, :, :] = img
                reference = reference_cube[0, :, :]
            elif (self.dstrMethod == 'running') & (i == 1):
                reference = reference_cube[0, :, :]
                img = self.read_speckle(self.spklList[i])
            elif (self.dstrMethod == 'running') & (i < self.dstrWindow):
                reference = np.nanmean(reference_cube[:i, :, :], axis=0)
                img = self.read_speckle(self.spklList[i])
            elif (self.dstrMethod == 'running') & (i > self.dstrWindow):
                reference = np.nanmean(reference_cube, axis=0)

            d_obj = Destretch(
                img,
                reference,
                kernels,
                return_vectors=True
            )
            dstr_im, dstr_vecs = d_obj.perform_destretch()
            if type(dstr_vecs) != list:
                dstr_vecs = [dstr_vecs]

            vecs_master = vecs_master + dstr_vecs

            if self.dstrMethod == 'running':
                reference_cube[int(i % self.dstrWindow), :, :] = dstr_im

            # Write FITS and vectors
            writeFile = os.path.join(self.dstrBase, str(i).zfill(5))
            np.save(writeFile + ".npy", vecs_master)

            fname = os.path.join(
                self.postSpeckleBase,
                self.dstrFilePattern.format(
                    self.date,
                    self.time,
                    i
                )
            )
            alpha = np.fromfile(alphaFlist[i], dtype=np.float32)[0]
            self.write_fits(fname, dstr_im, self.hdrList[i], alpha=alpha)

    def destretch_to_reference(self):
        """Destretch to a reference list of vectors"""
        # Step 0: Get list of files from self.dstrBase,
        #   which should be set up in the config file from self.referenceChannel
        # Step 1: From the list of destretch targets and the list of vectors, see if there's a dimension mis-match
        # Step 2: If there is a mismatch, divide the vector list len by the target len.
        #   Step 2.5: Use this to determine the vector list index by having a second iterable, and passing this iterable
        #   to round()
        # Step 3: Apply destretch
        # Step 4: Write FITS.
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

        self.dstrVectorList = sorted(
            glob.glob(
                os.path.join(
                    self.dstrBase,
                    '*.npy'
                )
            )
        )

        try:
            self.assert_flist(self.dstrVectorList)
        except AssertionError as err:
            print("Error: Vector List: {0}".format(err))
            raise

        if len(spklFlist) != len(self.dstrVectorList):
            dstr_vec_increment = len(spklFlist)/len(self.dstrVectorList)
        else:
            dstr_vec_increment = 1
        dstr_ctr = 0
        for i in tqdm(range(len(spklFlist)), desc="Appling Destretch Vectors..."):
            img = self.read_speckle(spklFlist[i])
            img = self.perform_bulk_translation(img)
            img = self.perform_fine_translation(img)
            vecs = np.load(self.dstrVectorList[int(round(dstr_ctr))])
            d = Destretch(
                img,
                img,
                self.kernels,
                warp_vectors=vecs
            )
            dstrim = d.perform_destretch()
            fname = os.path.join(
                self.postSpeckleBase,
                self.dstrFilePattern.format(
                    self.date,
                    self.time,
                    i
                )
            )
            alpha = np.fromfile(alphaFlist[i], dtype=np.float32)[0]
            self.write_fits(fname, dstrim, self.hdrList[i], alpha=alpha)
            dstr_ctr += dstr_vec_increment

    def write_fits(self, fname, data, hdr, alpha=None):
        """Write destretched FITS files."""
        allowed_keywords = ['DATE', 'EXPOSURE', 'HIERARCH']
        header = open(hdr, 'r').readlines()
        hdul = fits.HDUList(fits.PrimaryHDU(data))
        for i in range(len(header)):
            slug = header[i].split("=")[0]
            field = header[i].split("=")[-1].split("/")[0]
            if len(header[i].split("=")[-1].split("/")) == 1:
                field = field.split("\n")[0].strip()
            field = field.strip()[1:-1].strip()
            if any(substring in slug for substring in allowed_keywords):
                hdul[0].header[slug] = field
        hdul[0].header['AUTHOR'] = 'sellers'
        if alpha:
            hdul[0].header['SPKLALPHA'] = alpha
        hdul.writeto(fname, overwrite=True)

    def remove_flows(self):
        """Function to perform destretch on single channel, removing flows."""
        destretch_coord_list = self.dstrVectorList
        smooth_number = self.flowWindow
        if self.dstrMethod == 'reference':
            median_number = 10
        else:
            median_number = self.dstrWindow

        if self.kernels[0] == 0:
            index_in_file = 1
        else:
            index_in_file = 0

        template_coords = np.load(destretch_coord_list[0])
        # If the first list entry is bulk frame shifts, truncate them for now
        if len(template_coords[0].shape) < 3:
            template_coords = template_coords[1:]
        grid_x, grid_y = np.meshgrid(
            np.arange(template_coords.shape[2]),
            np.arange(template_coords.shape[1])
        )
        shifts_all = np.zeros(
            (
                template_coords[0].shape[0],
                template_coords[0].shape[1],
                template_coords[0].shape[2],
                len(destretch_coord_list)
            ),
            dtype=np.float32
        )
        shifts_bulk = np.zeros((2, len(destretch_coord_list)), dtype=np.float32)
        shifts_bulk_sum = np.zeros((2, len(destretch_coord_list)), dtype=np.float32)
        shifts_bulk_corr = np.zeros(
            (
                template_coords[0].shape[0],
                template_coords[0].shape[1],
                template_coords[0].shape[2],
                len(destretch_coord_list)
            ),
            dtype=np.float32
        )
        shifts_corr_sum = np.zeros(
            (
                template_coords[0].shape[0],
                template_coords[0].shape[1],
                template_coords[0].shape[2],
                len(destretch_coord_list)
            ),
            dtype=np.float32
        )
        for i in tqdm(range(len(destretch_coord_list)), desc='Calculating Shifts...'):
            coords = np.load(destretch_coord_list[i])[index_in_file].astype(np.float32)
            coords[coords == -1] = np.nan
            shifts_all[0, :, :, i] = coords[0, :, :] - grid_y
            shifts_all[1, :, :, i] = coords[1, :, :] - grid_x
            shifts_bulk[:, i] = np.nanmedian(shifts_all[:, :, :, i], axis=(1, 2))
            shifts_bulk_corr[0, :, :, i] = shifts_all[0, :, :, i] - shifts_bulk[0, i]
            shifts_bulk_corr[1, :, :, i] = shifts_all[1, :, :, i] - shifts_bulk[1, i]
            if i == 0:
                shifts_bulk_sum[:, i] = shifts_bulk[:, i]
                shifts_corr_sum[:, :, :, i] = shifts_bulk_corr[:, :, :, i]
            else:
                shifts_bulk_sum[:, i] = shifts_bulk_sum[:, i-1] + shifts_bulk[:, i]
                shifts_corr_sum[:, :, :, i] = shifts_corr_sum[:, :, :, i-1] + shifts_bulk_corr[:, :, :, i]

        pbar = tqdm(
            total=shifts_corr_sum.shape[0]*shifts_corr_sum.shape[1]*shifts_corr_sum.shape[2],
            desc='Removing Flows...'
        )
        flow_detr_shifts = np.zeros(shifts_corr_sum.shape)
        for y in range(shifts_corr_sum.shape[1]):
            for x in range(shifts_corr_sum.shape[2]):
                for cd in range(shifts_corr_sum.shape[0]):
                    disp_sequence = shifts_corr_sum[cd, y, x, :]
                    smoothed_sequence = scindi.uniform_filter1d(
                        scindi.median_filter(
                            disp_sequence, size=median_number, mode='nearest'
                        ),
                        smooth_number, mode='nearest'
                    )
                    flow_detr_shifts[cd, y, x, :] = disp_sequence - smoothed_sequence
                    pbar.update(1)
        pbar.close()
        for i in tqdm(range(len(destretch_coord_list)), desc='Adding jitter back...'):
            flow_detr_shifts[0, :, :, i] = flow_detr_shifts[0, :, :, i] + grid_y + shifts_bulk_sum[0, i]
            flow_detr_shifts[1, :, :, i] = flow_detr_shifts[1, :, :, i] + grid_x + shifts_bulk_sum[1, i]

        self.write_dstr_from_array(flow_detr_shifts, self.deflowDir)

    def write_dstr_from_array(self, array_to_write, writeDir):
        """General-purpose write numpy save file of destretch coordinates.
        Since the pyDestretch algorithm expects input coordinates as a list of arrays,
        we need to repack the arrays to write them."""
        for i in range(array_to_write.shape[-1]):
            writeFile = os.path.join(writeDir, str(i).zfill(5))
            np.save(writeFile + ".npy", [array_to_write[:, :, :, i]])

    def assert_flist(self, flist):
        assert (len(flist) != 0), "List contains no matches"

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

# Note to future Sean: Here's what Kevin Does.
# He works with the control points, NOT the full array of warped coordinates. This is good -- saves memory
# 1.) Remove bulk shifts from calculated destretch vecotrs.
#   These are full-frame shifts, which should've mostly been removed already.
#   He does this by estimating the shifts by subtracting the reference and measured control points,
#   And determining what trends affect every control point.
# 2.) Subtract these bulk trends from the control points
# 3.) For each bulk-subbed CONTROL POINT in the SEQUENCE of destretch images,
#   remove long-term (~minutes) trends by taking the rolling mean of each coordinate, and subtracting it from the actual
#   time series
# 4.) Add the bulk shifts back in (we still want them corrected)
# This will require a REWRITE of the pyDestretch class to take a sequence instead of a target and reference.
# It shouldn't be too bad -- we'll need to alter the target and reference to instead want a sequence of images
# Helper functions for the reading of these images (speckle vs fits)
# Control Point altering functions for removing the bulk and flow shifts, then adding bulk back in
# Extra params in the class initialization for the bulk and flow windows.

# NO. DO NOT REWRITE THE STRUCTURE OF THE CLASS.
# Why? Because if we do that, the class can only be used in conjunction with ssosoft or a config file,
# which is unflexible
# Instead, detrending flows should be done by initially destretching a sequence of images, returning the warp vectors,
# but not writing the detretched images. The returned coordinates should be saved somewhere on the disk, defined by the
# config file. Then, we can load in as needed, apply the detrend,
# and then apply the detrended coordinates to the raw images.
# If we return the shifts as well, we can pad the images, place them,
# and apply the transformation to them in one fell swoop.
# This has an added benefit of allowing the core destretch class to be used for any instrument.

# How do we do this?