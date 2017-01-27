import unittest
import os
from gdal_utils import gdal_utils as gu

datadir = os.path.dirname(os.path.realpath(__file__))
intif = os.path.join(datadir, "image.tif")
inshp = os.path.join(datadir, 'inshp.shp')
outtif = os.path.join(datadir, "image_copy.tif")

class TestUtils(unittest.TestCase):

    def test_retrieve_array_shape(self):
        image, _ = gu.retrieve_array(intif)
        self.assertEqual(image.shape, (5L, 205L, 198L))
        # cleanup
        image = None

    def test_save_same_format_shape(self):
        image, _ = gu.retrieve_array(intif)
        gu.save_same_format(intif, outtif, image)
        try:
            copy_image, _ = gu.retrieve_array(outtif)
            self.assertEqual(image.shape, copy_image.shape)
        finally:
            os.remove(outtif)

    def test_cutline(self):
        try:
            gu.cutline(intif, inshp, outtif)
            self.assertTrue(os.path.isfile(outtif))
        finally:
            try:
                os.remove(outtif)
            except OSError:
                pass

    def test_reproject(self):
        try:
            gu.warp_reproject(intif, t_srs='EPSG:4326', outfile=outtif)
            self.assertTrue(os.path.isfile(outtif))
        finally:
            try:
                os.remove(outtif)
            except OSError:
                pass

    def test_get_resolution(self):
        xres, yres = gu.get_resolution(intif)
        self.assertEqual(xres, 2)
        self.assertEqual(yres, 2)


class TestCMDs(unittest.TestCase):

    def test_cmd_gdalwarp_cutline(self):
        cmd = gu.cmd_gdalwarp_cutline(intif, inshp, outtif, preserve_resolution=True)
        self.assertEqual(cmd, [gu.gdalwarp_exe, '-tr', '2.0', '2.0', '-cutline', inshp, '-crop_to_cutline', '-dstnodata', '9.96920996839e+36', intif, outtif])
