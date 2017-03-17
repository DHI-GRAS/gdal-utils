import fiona
import shapely


def read_as_wkt(infile, feat=0):
    with fiona.open(infile) as c:
        feature = c[feat]
        if feature is None:
            raise IndexError('Collection has only {} features.'.format(len(c)))
    return feature_to_wkt(feature)


def feature_to_wkt(feature):
    g = feature['geometry']
    geometry = shapely.geometry.shape(g)
    return geometry.wkt
