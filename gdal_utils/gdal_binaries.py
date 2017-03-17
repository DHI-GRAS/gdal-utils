import os
import sys
import glob
import shutil
import tempfile
import warnings
import subprocess
import logging

if 'rasterio' not in sys.modules:
    from .gdal_utils import get_file_nodata
    from .gdal_utils import get_resolution
else:
    raise NotImplementedError('Rasterio detected. This is not yet supported.')

logger = logging.getLogger(__name__)


def find_gdal_exe(gdalcmd):
    if sys.platform.startswith('linux'):
        return gdalcmd
    try:
        if not gdalcmd.endswith('.exe'):
            gdalcmd += '.exe'
        pattern = os.path.join('C:\\', 'OSGeo4W*', 'bin', gdalcmd)
        cmdpath = glob.glob(pattern)[0]
    except IndexError:
        # no OSGeo4W installed
        cmdpath = gdalcmd
    return cmdpath


gdal_translate_exe = find_gdal_exe('gdal_translate')
gdal_rasterize_exe = find_gdal_exe('gdal_rasterize')
gdalwarp_exe = find_gdal_exe('gdalwarp')
gdalbuildvrt_exe = find_gdal_exe('gdalbuildvrt')


def _get_startupinfo():
    """startupinfo to suppress external command windows"""
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return startupinfo


def run_cmd(cmd, outfile):
    subprocess.check_output(cmd,
            stdin=subprocess.PIPE, stderr=subprocess.PIPE,  # to avoid error in pythonw
            startupinfo=_get_startupinfo())
    if not os.path.isfile(outfile):
        cmdstr = subprocess.list2cmdline(cmd)
        raise RuntimeError('GDAL command failed. No output created with '
                'cmd \'{}\'.'.format(cmdstr))


def cmd_gdalwarp_cutline(intif, inshp, outtif, preserve_resolution=True, extra=[]):
    if os.path.isfile(outtif):
        os.remove(outtif)
    dstnodata = get_file_nodata(intif)
    cmd = [gdalwarp_exe]
    if preserve_resolution and '-tr' not in extra:
        xres, yres = get_resolution(intif)
        cmd += ['-tr', str(xres), str(yres)]
    cmd += ['-cutline', inshp]
    cmd += ['-crop_to_cutline']
    cmd += ['-dstnodata', str(dstnodata)]
    cmd += extra
    cmd += [intif, outtif]
    return cmd


def cmd_gdalwarp_reproject(infile, t_srs, outfile, r='near'):
    dstnodata = get_file_nodata(infile)
    cmd = [gdalwarp_exe]
    cmd += ['-t_srs', t_srs]
    cmd += ['-r', r]
    cmd += ['-dstnodata', str(dstnodata)]
    cmd += [infile, outfile]
    return cmd


def cmd_gdalwarp_reproject_cutline(intif, inshp, outtif, t_srs, r='near',
        preserve_resolution=True, extra=[]):
    dstnodata = get_file_nodata(intif)
    cmd = [gdalwarp_exe]
    if preserve_resolution and '-tr' not in extra:
        xres, yres = get_resolution(intif)
        cmd += ['-tr', str(xres), str(yres)]
    cmd += ['-t_srs', t_srs]
    cmd += ['-r', r]
    cmd += ['-cutline', inshp]
    cmd += ['-crop_to_cutline']
    cmd += ['-dstnodata', str(dstnodata)]
    cmd += extra
    cmd += [intif, outtif]
    return cmd


def cmd_gdalbuildvrt(infiles, outfile, resolution='average', separate=False, proj_difference=False, extra=[]):
    cmd = [gdalbuildvrt_exe]
    cmd += ['-q']
    if resolution != 'average':
        cmd += ['-resolution', resolution]
    if separate:
        cmd += ['-separate']
    if proj_difference:
        cmd += ['-allow_projection_difference']
    cmd += extra
    cmd += [outfile]
    cmd += infiles
    return cmd


def cmd_gdal_translate(infile, outfile,
        of='GTiff', ot='',
        lonlim=(), latlim=(),
        extent=[], outsize=None,
        co_dict={}, extra=[]):
    """Generate gdal_translate command

    Parameters
    ----------
    infile, outfile : str
        paths to in/out files
    of : str
        output format
    ot : str
        output data type
    extent : list or str
        extent to subset
        NB: extent is (ulx uly lrx lry)
    outsize : list of len 2
        output size
    co_dict : dict
        creation options
        e.g. {'COMPRESS': 'JPEG', 'JPEG_QUALITY': 75}
    extra : list
        extra commands
    """
    cmd = [gdal_translate_exe]
    if of.lower() != 'gtiff':
        cmd += ['-of', of]
    if ot:
        cmd += ['-ot', ot]
    if lonlim and latlim:
        # (ulx uly lrx lry)
        extent = [lonlim[0], latlim[1], lonlim[1], latlim[0]]
    if extent:
        if ',' in extent:
            extent = extent.split(',')
        cmd += ['-projwin'] + [str(e) for e in extent]
    if outsize and outsize not in [['100%', 0], ['100%', '100%']]:
        cmd += ['-outsize'] + [str(s) for s in outsize]
    if co_dict:
        for k, v in co_dict.items():
            cmd += ['-co', '{}={}'.format(k, v)]
    cmd += extra
    cmd += [infile, outfile]
    return cmd


def warp_reproject(infile, t_srs, outfile, **kwargs):
    """gdalwarp

    Parameters
    ----------
    infile, outfile : str
        paths to input and output files
    **kwargs : dict
        passed to cmd_gdalwarp_reproject
    """
    cmd = cmd_gdalwarp_reproject(infile, t_srs, outfile, **kwargs)
    run_cmd(cmd, outfile)


def buildvrt(infiles, outfile, **kwargs):
    """GDAL build virtual raster

    Parameters
    ----------
    infiles : list of str
        paths to input files
    outfile : str
        path to output vrt
    kwargs : dict
        keyword arguments passed to
        cmd_gdalbuildvrt
    """
    cmd = cmd_gdalbuildvrt(infiles, outfile, **kwargs)
    run_cmd(cmd, outfile)


def translate(infile, outfile, **kwargs):
    """GDAL build virtual raster

    Parameters
    ----------
    infile : str
        path to input file
    outfile : str
        path to output file
    kwargs : dict
        keyword arguments passed to
        cmd_gdal_translate
    """
    cmd = cmd_gdal_translate(infile, outfile, **kwargs)
    run_cmd(cmd, outfile)


def cutline(intif, inshp, outtif=None, t_srs=None, **kwargs):
    """GDAL gdalwarp cutline with default parameters

    Parameters
    ----------
    intif : str
        path to input tif
    inshp : str
        path to input shape file
    outtif : str or None
        path to output tif
        if not defined, intif will be overwritten
    t_srs : str
        t_srs
    kwargs : dict
        arguments passed to cmd_gdalwarp_cutline
        of cmd_gdalwarp_reproject_cutline (if t_srs is defined)
    """
    inplace = False
    tempdir = None
    if outtif is None or outtif == intif:
        tempdir = tempfile.mkdtemp()
        outtif = os.path.join(tempdir, 'temp.tif')
        inplace = True
    # run gdalwarp cutline
    if t_srs:
        cmd = cmd_gdalwarp_reproject_cutline(intif, inshp, outtif, t_srs, **kwargs)
    else:
        cmd = cmd_gdalwarp_cutline(intif, inshp, outtif, **kwargs)
    run_cmd(cmd, outtif)
    # move temp file to infile for in-place
    if inplace:
        shutil.copyfile(outtif, intif)
        try:
            shutil.rmtree(tempdir)
        except OSError:
            warnings.warn('Temporary files were not removed from \'{}\'.'.format(tempdir))
            pass


def gdal_compress(infile, outfile, compress, extra=''):
    """Use gdal_translate to save a compressed version of the file
       preserving nodata

    Parameters
    ----------
    infile, outfile : str
        input and output files
        outfile will be overwritten if exists
    compress : str
        e.g. LZW or DEFLATE
    extra : str
        extra flags to gdal_translate
    """
    if os.path.exists(outfile):
        os.remove(outfile)
    src_nodata = get_file_nodata(infile)
    cmd = gdal_translate_exe
    cmd += ' -co COMPRESS={} -a_nodata {} {} {} {}'.format(compress, src_nodata, extra, infile, outfile)
    subprocess.call(cmd)
    if not os.path.isfile(outfile):
        raise RuntimeError('GDAL compression not successful with command \'{}\'.'.format(cmd))
