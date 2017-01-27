import unittest
import os
import glob
from graspy import ogr_utils as ou

datadir = os.path.dirname(os.path.realpath(__file__))
intif = os.path.join(datadir, 'image.tif')
inshp = os.path.join(datadir, 'inshp.shp')
outshp = os.path.join(datadir, 'outshp.shp')
outtif = os.path.join(datadir, 'image_copy.tif')

def _cleanup_shp(shpfile):
    pattern = os.path.splitext(shpfile)[0] + '.*'
    for fname in glob.glob(pattern):
        try:
            os.remove(fname)
        except OSError:
            pass

class TestOGRUtils(unittest.TestCase):

    def test_buffer(self):
        try:
            ou.buffer_shapefile(inshp, outshp, bufsize=-10)
            self.assertTrue(os.path.isfile(outshp))
        finally:
            _cleanup_shp(outshp)

    def test_get_wkt(self):
        wkt = ou.shp2wkt(inshp)
        self.assertTrue(wkt, 'Polygon')


if __name__ == '__main__':
    unittest.main()
