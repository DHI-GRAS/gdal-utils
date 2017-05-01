import os
import shutil
import logging
import tempfile

import gdal
import osr
import numpy as np
import sys
import ogr

from .gdal_scripts import gdal_merge as gdal_merge_py

logger = logging.getLogger('gdal_utils.gdal_utils')


dtype_np_to_gdal = {
        'i2': gdal.GDT_Int16,
        'i4': gdal.GDT_Int32,
        'u1': gdal.GDT_Byte,
        'u2': gdal.GDT_UInt16,
        'f4': gdal.GDT_Float32,
        'f8': gdal.GDT_Float64}

default_nodata = {
        'f4': 9.969209968386869e+36,
        'f8': 9.969209968386869e+36,
        'i1': -127,
        'i2': -32767,
        'i4': -2147483647,
        'i8': -9223372036854775806L,
        'u1': 255,
        'u2': 65535,
        'u4': 4294967295L,
        'u8': 18446744073709551614L}

# Pixel data types
# http://www.gdal.org/gdal_8h.html#a22e22ce0a55036a96f652765793fb7a4
gdt_dtype = ['', 'u1', 'u2', 'i2', 'u4', 'i4', 'f4', 'f8']

class GdalErrorHandler():
    def __init__(self):
        self.err_level = gdal.CE_None
        self.err_no = 0
        self.err_msg = ''

    def handler(self, err_level, err_no, err_msg):
        self.err_level = err_level
        self.err_no = err_no
        self.err_msg = err_msg


class gdal_handle_errors:
    def __init__(self):
        pass

    def __enter__(self):
        err = GdalErrorHandler()
        handler = err.handler
        gdal.PushErrorHandler(handler)
        gdal.UseExceptions()

    def __exit__(self, type, value, traceback):
        gdal.PopErrorHandler()


def get_default_nodata(dtype):
    dt = np.dtype(dtype)
    key = dt.str[1:]
    return np.array(default_nodata[key], dt)[()]


def get_gdal_dtype(dtype):
    dt = np.dtype(dtype)
    key = dt.str[1:]
    return dtype_np_to_gdal[key]


def get_file_nodata(intif):
    """Get nodata from first band in file"""
    with gdal_open(intif) as ds:
        b = ds.GetRasterBand(1)
        src_nodata = b.GetNoDataValue()
        if src_nodata is None:
            key = gdt_dtype[b.DataType]
            src_nodata = default_nodata[key]
    return src_nodata


def check_gdal_success(outfile, cmd):
    """Make sure GDAL command `cmd` succeeded in creating `outfile`"""
    if not os.path.isfile(outfile):
        raise RuntimeError('GDAL command \'{}\' did not produce the '
                'expected output file {}.'.format(cmd, outfile))


def retrieve_array(infile, iband=None, tgt_dtype='float32'):
    """Retrieve the band data from a GeoTif as filled array"""
    a_masked = retrieve_array_masked(infile, iband, tgt_dtype)
    return a_masked.filled(), a_masked.fill_value


def retrieve_array_masked(infile, iband=None, tgt_dtype=None):
    """Retrieve the band data from a GeoTif as masked array"""
    with gdal_open(infile) as ds:
        return get_array(ds, iband, tgt_dtype)


def get_array(ds, iband=None, tgt_dtype=None, tgt_nodata=None):
    """Returns a masked array from open GDAL dataset

    Parameters
    ----------
    ds : gdal.Open
        Dataset
    iband : int
        1-based band number
        default: all bands
    tgt_dtype : dtype
        convert to this dtype
        default: same as input
    tgt_nodata : float, int
        target no-data values
        default: matches tgt_dtype
    """
    if iband is None:
        # read all bands
        a = ds.ReadAsArray()
        src_nodata = ds.GetRasterBand(1).GetNoDataValue()
    else:
        band = ds.GetRasterBand(iband)
        a = band.ReadAsArray()
        src_nodata = ds.GetRasterBand(1).GetNoDataValue()

    if src_nodata is not None:
        if a.dtype == 'f':
            # floating point comparison
            mask = np.abs(a - src_nodata) < 1e-10  # np.isclose uses way too much memory
        else:
            # integer
            mask = a == src_nodata
    else:
        try:
            mask = ~np.isfinite(a)
        except TypeError:
            mask = False
    tgt_dtype = tgt_dtype or str(a.dtype)
    tgt_nodata = tgt_nodata if tgt_nodata is not None else get_default_nodata(tgt_dtype)
    a_masked = np.ma.masked_where(mask, a, copy=False)
    a_masked.set_fill_value(tgt_nodata)
    return a_masked.astype(tgt_dtype)


def make_gdalstr(fname, group=None, varn=None):
    if fname.lower().endswith('.nc') and varn is not None:
        gdalstr = 'NETCDF:{}:{}'.format(fname, varn)
    if fname.lower().endswith('.hdf') and varn is not None and group is not None:
        gdalstr = 'HDF4_EOS:EOS_GRID:{}:{}:{}'.format(fname, group, varn)
    else:
        gdalstr = fname
    return gdalstr


class gdal_open:

    def __init__(self, fname, mode=gdal.GA_ReadOnly, group=None, varn=None):
        """Open file with GDAL (tif or netCDF)

        Parameters
        ----------
        fname : str
            path to input file
            can be netCDF
        group : str
            name of group in netCDF or HDF file
        varn : str, optional
            name of variable in netCDF or HDF file
        """
        gdalstr = make_gdalstr(fname, group=group, varn=varn)
        self.file = gdal.Open(gdalstr, mode)
        if self.file is None:
            raise IOError(
                "Loading file '{}' with GDAL failed".format(gdalstr))

    def __enter__(self):
        return self.file

    def __exit__(self, type, value, traceback):
        self.file = None


def add_crs(outfile, projection, geotransform):
    """Assign projection and geotransform to dataset"""
    with gdal_open(outfile, mode=gdal.GA_Update) as ds:
        if geotransform and geotransform != (0,1,0,0,0,1):
            ds.SetGeoTransform(geotransform)
        if projection:
            ds.SetProjection(projection)


def apply_mask_to_gtiff(infile, maskfile, outfile, maskvalue=None,
        tgt_dtype=None, tgt_nodata=None):
    """Apply mask from one GTiff to another, saving to a third

    Parameters
    ----------
    infile : str
        path to input file
    maskfile : str
        path to mask file
    outfile : str
        path to output file
    maskvalue : int, float, ...
        value in maskfile to use as mask
        if None, the nodata value of maskfile
        is used
    """
    # get data array and projection parameters
    with gdal_open(infile) as ds:
        arr = get_array(ds, iband=None, tgt_dtype=tgt_dtype, tgt_nodata=tgt_nodata)
        projkw = dict(
                geotransform=ds.GetGeoTransform(),
                projection=ds.GetProjection())

    # get mask array and extract mask
    maskarr = retrieve_array_masked(maskfile, iband=None)
    mask = maskarr.mask.copy()
    if maskvalue is not None:
        mask |= maskarr.filled() == maskvalue

    # mask values in array
    dimerr = ValueError('Array to mask must be 3D (band, j, i) or 2D (j, i).')
    if np.ndim(mask) == 2:
        if np.ndim(arr) == 2:
            arr[mask] = np.ma.masked
        elif np.ndim(arr) == 3:
            arr[:,mask] = np.ma.masked
        else:
            raise dimerr
    elif np.ndim(mask) == 3:
        if np.ndim(arr) == 3:
            arr[mask] = np.ma.masked
        else:
            raise dimerr
    else:
        raise dimerr

    return array_to_gtiff(arr, outfile, tgt_nodata=tgt_nodata, **projkw)


def array_to_gtiff(arr, outfile, projection, geotransform, banddim=0,
        tgt_nodata=None, create_options=['COMPRESS=LZW', 'BIGTIFF=IF_SAFER'], dtype=None):
    """
    Save a numpy array to GTiff

    Parameters
    ----------
    arr: ndarray
        a (n, m, b) matrix, where b is the band number,
        n and m is the row and collum size of the matrices.
    outfile: str
        filename
    projection: str
        image projection
    geotransform:
        image geotransform
    banddim:
        swapping of axis if (n, m, b) is not in the correct order
    tgt_nodata: float, int
        target nodata value
    create_options:
        passed to gdal.Create
    dtype: 9
        Sets the output file to the desied dtype

    Returns
    -------
    gdal.Dataset
    """
    err = GdalErrorHandler()
    handler = err.handler
    gdal.PushErrorHandler(handler)
    gdal.UseExceptions()
    if dtype is None:
        dtype = arr.dtype
    gdal_dtype = get_gdal_dtype(dtype)

    if tgt_nodata is None:
        tgt_nodata = getattr(arr, 'fill_value', get_default_nodata(dtype))
    # GDAL for some reason fails with other types
    # TODO: Find and resolve this issue
    tgt_nodata = int(tgt_nodata)

    arr = np.ma.masked_invalid(arr).filled(tgt_nodata)

    # get array into right format
    if np.ndim(arr) == 3:
        if banddim == 1:
            arr = np.swapaxes(arr, 0, 1)
            arr = np.swapaxes(arr, 1, 2)
        if banddim == 2:
            arr = np.swapaxes(arr, 1, 2)
            arr = np.swapaxes(arr, 0, 1)
    elif np.ndim(arr) == 2:
        arr = arr[np.newaxis, :, :]
        banddim = 0
    else:
        raise NotImplementedError("Need at least 2D data.")
    nbands, nx, ny = arr.shape

    if outfile == 'MEM':
        drv = gdal.GetDriverByName('MEM')
    else:
        drv = gdal.GetDriverByName('GTiff')

    out_tif = drv.Create(outfile, ny, nx, nbands, gdal_dtype, create_options)
    if out_tif is None:
        raise IOError('Unable to create new dataset in {}.'.format(outfile))

    try:
        out_tif.SetGeoTransform(geotransform)
        out_tif.SetProjection(projection)
        for b in range(nbands):
            out_tif.GetRasterBand(b+1).WriteArray(arr[b, :, :])
        out_tif.GetRasterBand(1).SetNoDataValue(tgt_nodata)
    finally:
        if outfile != 'MEM':
            out_tif = None
        gdal.PopErrorHandler()

    return out_tif


def save_same_format(samplefile, outfile, array, tgt_nodata=None, **kwargs):
    """Save an array to geoTiff copying format from input file"""
    kw = kwargs.copy()
    with gdal_open(samplefile) as ds:
        kw.update(
                geotransform=ds.GetGeoTransform(),
                projection=ds.GetProjection())
    return array_to_gtiff(array, outfile, tgt_nodata=tgt_nodata, **kw)


def dump_gtiff(dsmem, outfile):
    """Dump MEM dataset to GeoTIFF outfile"""
    drv = gdal.GetDriverByName('GTiff')
    drv.CreateCopy(outfile, dsmem)


def set_nodata(infile, outfile, src_nodata=None):
    """Set nodata value on infile and save as outfile"""
    data = retrieve_array_masked(infile, iband=None)
    data[data == src_nodata] = np.ma.masked
    save_same_format(infile, outfile, data)


def set_nodata_inplace(infile, src_nodata=None):
    """Set value to nodata on infile "in-place"

    Parameters
    ----------
    infile : str
        path to tif file to modify
    src_nodata : int, float
        value to mask
    """
    tempdir = tempfile.mkdtemp()
    try:
        tempcopy = os.path.join(tempdir, 'temp_copy.tif')
        tempnodata = os.path.join(tempdir, 'temp_nodata.tif')
        shutil.copy2(infile, tempcopy)
        set_nodata(tempcopy, tempnodata)
        shutil.copy(tempnodata, infile)
    finally:
        try:
            shutil.rmtree(tempdir)
        except OSError:
            logger.warn(
                    'Temporary directory could not be removed. '
                    'Remove manually: \'{}\'.'.format(tempdir))


def create_empty_like(tif_template, outfile, tgt_dtype, tgt_nodata=None):
    """Create empty GDAL dataset from template"""
    if tgt_nodata is None:
        tgt_nodata = get_default_nodata(tgt_dtype)

    gdal_dtype = get_gdal_dtype(tgt_dtype)

    # GDAL for some reason fails with other types
    # TODO: Find and resolve this issue
    tgt_nodata = int(tgt_nodata)
    with gdal_handle_errors():

        # get parameters
        with gdal_open(tif_template) as ds:
            nx = ds.RasterXSize
            ny = ds.RasterYSize
            nbands = ds.RasterCount
            geotransform = ds.GetGeoTransform()
            projection = ds.GetProjection()

        # create output file
        if outfile == 'MEM':
            drv = gdal.GetDriverByName('MEM')
        else:
            drv = gdal.GetDriverByName('GTiff')

        out = drv.Create(outfile, ny, nx, nbands, gdal_dtype)
        if out is None:
            raise IOError('Unable to create new dataset in {}.'.format(outfile))

        out.SetGeoTransform(geotransform)
        out.SetProjection(projection)
        band = out.GetRasterBand(1)
        band.Fill(tgt_nodata)

    if outfile == 'MEM':
        return out
    else:
        out = None


def warp_reproject_py(infile, outfile, t_epsg=4326, r='near'):

    # Define target SRS
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(t_epsg)
    dst_wkt = dst_srs.ExportToWkt()

    # error threshold
    error_threshold = 0.125  # error threshold --> use same value as in gdalwarp

    # resampling
    if r == 'near':
        resampling = gdal.GRA_NearestNeighbour
    elif r == 'bilinear':
        resampling = gdal.GRA_Bilinear
    else:
        raise ValueError('Resampling `r` should be \'near\' or \'bilinear\'.')

    with gdal_open(infile) as src_ds:
        # Call AutoCreateWarpedVRT() to fetch default values for target raster dimensions and geotransform
        tmp_ds = gdal.AutoCreateWarpedVRT(src_ds,
                                          None,  # src_wkt
                                          dst_wkt,
                                          resampling,
                                          error_threshold)
    # Create the final warped raster
    if outfile.lower().endswith('.vrt'):
        driver = 'VRT'
    else:
        driver = 'GTiff'
    gdal.GetDriverByName(driver).CreateCopy(outfile, tmp_ds)
    check_gdal_success(outfile, 'gdalwarp')


def cutline_to_shape_name(intif, inshp, t_srs=None):
    """
    Cut the image with shape file givng the image the extention of the shape name
    if intif is a string open it, if intif is a gdal.Dataset then find the name of the path

    Warning
    -------
    This function makes some terrible assumptions about existing directories.
    """
    # TODO: Move this to Bathy. It should never be used elsewhere.
    from . import gdal_binaries
    # test if it is a gdal image or a string the are used
    try:
        # if gdal image, the close it and use the file name
        if intif.GetFileList() is None:
            data = np.zeros((intif.RasterCount, intif.RasterYSize, intif.RasterXSize), dtype=float)
            for band in xrange(intif.RasterCount):
                data[band, :, :] = intif.GetRasterBand(band + 1).ReadAsArray()
            img_path = os.path.join(
                    os.path.dirname(os.path.dirname(inshp)),
                    'Input', 'tmp.tif')
            array_to_gtiff(data, img_path, intif.GetProjection(), intif.GetGeoTransform())
        else:
            img_path = intif.GetFileList()[0]
        # if no file name is pressent then make a save it to tmp
        intif = None
    except AttributeError:
        img_path = intif

    shape_name = os.path.splitext(os.path.basename(inshp))[0]
    outtif = os.path.join(os.path.dirname(img_path), shape_name + '.tif')
    gdal_binaries.cutline(img_path, inshp, outtif, t_srs)
    return gdal.Open(outtif)


def gdal_merge(infiles, outfile, cmd_extra=[]):
    """Gdal merge
    cmd_extra : e.g '-separate, -of HFA, -co, COMPRESSED=YES'   """
    cmd = []
    cmd += ['-o', outfile]
    cmd += cmd_extra
    cmd += infiles
    gdal_merge_py.main(cmd)
    check_gdal_success(outfile, cmd)


def get_resolution(infile):
    """Get the resolution of the input file in projected coordinates"""
    with gdal_open(infile) as ds:
        gt = ds.GetGeoTransform()
        return abs(gt[1]), abs(gt[5])


def world2Pixel(geoMatrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    pixel = int(round((x - ulX) / xDist))
    line = int(round((ulY - y) / xDist))
    return pixel, line


def calcMinCoveringExtent(imgA, imgB):
    """
    Input: two GDALDataset
    Output: minimum covering extent of the two datasets
    [minX, maxY, maxX, minY] (UL, LR)
    """
    aGeoTrans = imgA.GetGeoTransform()
    bGeoTrans = imgB.GetGeoTransform()
    minX = max(aGeoTrans[0], bGeoTrans[0])
    maxY = min(aGeoTrans[3], bGeoTrans[3])
    maxX = min(aGeoTrans[0] + imgA.RasterXSize*aGeoTrans[1], bGeoTrans[0] + imgB.RasterXSize*bGeoTrans[1])
    minY = max(aGeoTrans[3] + imgA.RasterYSize*aGeoTrans[5], bGeoTrans[3] + imgB.RasterYSize*bGeoTrans[5])
    return [minX, maxY, maxX, minY]


def rasterize(in_vector, out_raster='MEM', pixel_size=25):
    # Define pixel_size and NoData value of new raster
    NoData_value = np.nan

    # Filename of the raster Tiff that will be created
    raster_fn = out_raster

    # Open the data source and read in the extent
    source_ds = in_vector
    source_layer = source_ds.GetLayer()
    x_min, x_max, y_min, y_max = source_layer.GetExtent()

    # Create the destination data source
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)
    if out_raster != 'MEM':
        target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, x_res, y_res, 1, gdal.GDT_Byte)
    else:
        target_ds = gdal.GetDriverByName('MEM').Create(raster_fn, x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)

    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])

    source_ds = None
    return target_ds


def clipRasterWithShape(rasterImg, shapeImg):
    """Cutline returning MEM"""
    # get the raster pixels size and extent
    # rasterImg = gdal.Open(raster,gdal.GA_ReadOnly)
    rasterGeoTrans = rasterImg.GetGeoTransform()

    # rasterize the shape and get its extent
    shapeRasterImg = rasterize(shapeImg, 'MEM', rasterGeoTrans[1])
    shapeRasterGeoTrans = shapeRasterImg.GetGeoTransform()

    # make sure that raster and shapeRaster pixels are co-aligned
    ulX = rasterGeoTrans[0] + round((shapeRasterGeoTrans[0] - rasterGeoTrans[0]) / rasterGeoTrans[1]) * rasterGeoTrans[
        1]
    ulY = rasterGeoTrans[3] + round((shapeRasterGeoTrans[3] - rasterGeoTrans[3]) / rasterGeoTrans[5]) * rasterGeoTrans[
        5]
    shapeRasterGeoTrans = (
            ulX, shapeRasterGeoTrans[1], shapeRasterGeoTrans[2],
            ulY, shapeRasterGeoTrans[4], shapeRasterGeoTrans[5])

    # get minimum covering extent for the raster and the shape and covert them
    # to pixels
    minX, maxY, maxX, minY = calcMinCoveringExtent(rasterImg, shapeRasterImg)
    rasterSubsetPixs = world2Pixel(rasterGeoTrans, minX, maxY) + world2Pixel(rasterGeoTrans, maxX, minY)
    shapeRasterSubsetPixs = world2Pixel(shapeRasterGeoTrans, minX, maxY) + world2Pixel(shapeRasterGeoTrans, maxX, minY)

    # clip the shapeRaster to min covering extent
    shapeRasterData = shapeRasterImg.GetRasterBand(1).ReadAsArray()
    shapeRasterClipped = shapeRasterData[shapeRasterSubsetPixs[1]:shapeRasterSubsetPixs[3],
                         shapeRasterSubsetPixs[0]:shapeRasterSubsetPixs[2]]
    mask = shapeRasterClipped > 0

    if shapeRasterClipped.size == 0 or shapeRasterClipped[mask].size == 0:
        sys.exit("\nThe shafile is not covering the area of you image: " +
                 "\nAre you using the correct shapefile?")

    # go through the raster bands, clip the to the minimum covering extent and mask out areas not covered by vector
    maskedData = np.zeros((np.shape(shapeRasterClipped)[0], np.shape(shapeRasterClipped)[1], rasterImg.RasterCount))
    for band in range(rasterImg.RasterCount):
        rasterData = rasterImg.GetRasterBand(band + 1).ReadAsArray()
        clippedData = rasterData[rasterSubsetPixs[1]:rasterSubsetPixs[3], rasterSubsetPixs[0]:rasterSubsetPixs[2]]
        maskedData[:, :, band] = np.where(mask, clippedData, 0)

    # get the geotransform array for the masekd array
    maskedGeoTrans = (ulX, rasterGeoTrans[1], rasterGeoTrans[2], ulY, rasterGeoTrans[4], rasterGeoTrans[5])

    # save the masked img to memory and return it
    return array_to_gtiff(maskedData, "MEM", rasterImg.GetProjection(), maskedGeoTrans, banddim=2)


def openAndClipRaster(inFilename, shapeRoiFilename):
    """Like cutline but returns MEM"""
    with gdal_open(inFilename) as inImg:
        # unclipped image
        if not shapeRoiFilename or not os.path.exists(shapeRoiFilename):
            return inImg

        shapeRoi = ogr.Open(shapeRoiFilename)
        clippedImg = clipRasterWithShape(inImg, shapeRoi)

        shapeRoi = None, None
    return clippedImg


def buffer_extent(extent, geotransform):
    """Buffer extent by one pixel to the east and south (if necessary) to make
       sure that gdal_translate leaves no bit of AOI out when subsetting

    Parameters
    ----------
    Extent is minX, maxX, minY, maxY
    Geotransform is GDAL geotransform
    """
    e = extent
    # From what I could figure out gdal_translate first shifts the extent north-west until UL corner aligns
    # with a UL corner of a pixel in the source layer. Then the BR corner is rounded to the BR corner of
    # the closest pixel. So if the north(west) pixel is closer then south(east) then the clipped layer will
    # not include all of the area specified in the extent.
    shiftWest = float(e[0]) % geotransform[1]
    if 0 < float(e[1]) % geotransform[1] / geotransform[1] < 0.5:
        east = float(e[1]) + geotransform[1]
    else:
        east = float(e[1])
    e[1] = str(east + shiftWest)
    shiftNorth = float(e[3]) % geotransform[5]
    if 0 < float(e[2]) % geotransform[5] / geotransform[5] < 0.5:
        south = float(e[2]) + geotransform[5]
    else:
        south = float(e[2])
    e[2] = str(south + shiftNorth)
    return e
