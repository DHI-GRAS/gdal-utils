import ogr

def buffer_shapefile(inshp, outshp, bufsize):
    """Buffer a shapefile

    Parameters
    ----------
    inshp : str
        path to input shape file
    outshp : str
        path to output shape file
    bufsize : int, float
        buffer size in projection units
    """
    shp = ogr.Open(inshp)
    if shp is None:
        raise IOError('Unable to read shape file \'{}\'.'.format(inshp))

    drv = shp.GetDriver()
    drv.CopyDataSource(shp, outshp)
    shp.Destroy()

    buf = ogr.Open(outshp, 1)
    try:
        lyr = buf.GetLayer(0)

        for i in xrange(lyr.GetFeatureCount()):
                feat = lyr.GetFeature(i)
                lyr.DeleteFeature(i)
                geom = feat.GetGeometryRef()
                feat.SetGeometry(geom.Buffer(bufsize))
                lyr.CreateFeature(feat)
    finally:
        buf.Destroy()


def extent_from_shpfile(inshp):
    """Get extent string from shapefile"""
    drv = ogr.GetDriverByName('ESRI Shapefile')
    ds = drv.Open(inshp)
    if ds is None:
        raise IOError('Unable to read shape file \'{}\'.'.format(inshp))
    try:
        layer = ds.GetLayer()
        extentShp = layer.GetExtent()
        return str(extentShp)[1:-1].replace(' ','')
    finally:
        ds.Destroy()


def shp2wkt(inshp):
    """Get POLYGON ((X Y ...)) string for shapefile"""
    drv = ogr.GetDriverByName('ESRI Shapefile')
    ds = drv.Open(inshp)
    if ds is None:
        raise IOError('Unable to read shape file \'{}\'.'.format(inshp))
    try:
        layer = ds.GetLayer()
        feature = layer.GetFeature(0)
        geometry = feature.geometry()
        return geometry.ExportToWkt()
    finally:
        ds.Destroy()


class ogr_open:

    def __init__(self, fname):
        """ogr.Open with check
        """
        self.file = ogr.Open(fname)
        if self.file is None:
            raise IOError(
                "Loading file '{}' with OGR failed".format(fname))

    def __enter__(self):
        return self.file

    def __exit__(self, type, value, traceback):
        self.file = None
