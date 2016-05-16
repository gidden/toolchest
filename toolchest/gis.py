import warnings

import numpy as np
import rasterio as rio
import fiona as fio

from itertools import product
from matplotlib import pyplot as plt
from six import string_types


def collapse_raster(raster, factor, src_transform, density=False):
    """Turn a high-detailed raster into a lower-detailed version, combining all values

    Parameters
    ----------
    raster : np.array or similar
    factor : int, e.g., use 4 to turn a 1/8th-degree raster into 1/2 degree
    src_trasform : affine.Affine or similar
    density : bool, whether the raster value is a density measure (and thus
    should be normalized by area)

    Returns
    -------
    ary : np.array for new raster
    to_transform : rasterio Affine for new raster

    """
    raster = np.copy(raster)
    raster[np.isnan(raster)] = 0  # make nans into something that is addable
    ret = sum_chunks(sum_chunks(raster, factor), factor, axis=0)
    if density:
        ret /= factor ** 2
    t = src_transform._asdict()
    t['a'] *= factor
    t['e'] *= factor
    dst_affine = rio.Affine(*t.values()[:6])
    return ret, dst_affine


def raster_area(ny, transform):
    """Returns an array of a given shape defining geospatial area.

    Parameters
    ----------
    ny : int, the number of cells in the latitudinal (y) dimension
    transform: affine.Affine or similar; if a list is provided it is assumed
    that it is ordered as a GDAL transform

    Returns
    -------
    area : a 1-D column vector describing the area of each latitudinal cell in
    km^2

    Example
    -------
    >>> with rasterio.open('raster.nc') as src:
    >>>     transform = src.affine
    >>>     raster = src.read()[0]
    >>> area = gis_tools.area(raster.shape[0], transform)
    >>> normalized = raster / area

    """
    radius = 6371  # authalic earth radius in km

    t = transform._asdict()
    diff_lat = t['e']
    diff_lon = t['a']
    top = t['f']  # top lat
    bottom = top + diff_lat * ny  # bottom lat

    rads = np.linspace(bottom, top, num=ny + 1) * np.pi / 180
    area = np.array([np.sin(rads[i + 1]) - np.sin(rads[i])
                     for i in range(len(rads) - 1)]) * diff_lon * radius ** 2

    return area.reshape(ny, 1)


def spread(a, shape):
    """Returns a new array with the given shape that has values equal to the
    original array spread to cover the given shape, e.g.:

    >>> a = np.arange(1, 5).reshape(2, 2)
    >>> spread(a, (4, 4))
    >>> array([[ 1.,  1.,  2.,  2.],
    >>>        [ 1.,  1.,  2.,  2.],
    >>>        [ 3.,  3.,  4.,  4.],
    >>>        [ 3.,  3.,  4.,  4.]])
    """
    x = np.zeros(shape)
    rspan = shape[0] / a.shape[0]
    cspan = shape[1] / a.shape[1]
    for r, c in product(range(rspan), range(cspan)):
        x[r::rspan, c::cspan] = a
    return x


def plot_raster(raster, bidx=1, nodata=None, func=None, save=None):
    """Plots a raster (file or array), converting `nodata` to `nan`s."""
    if isinstance(raster, string_types):
        with rio.open(raster) as src:
            raster = src.read(bidx)
            nodata = nodata or src.nodata
    if nodata is not None:
        raster = raster.astype(np.float)
        raster[raster == nodata] = np.nan
    raster = raster if func is None else func(raster)
    plt.imshow(raster, cmap=plt.cm.viridis)
    if save is not None:
        plt.savefig(save)
    plt.colorbar()
    plt.show()


def write_raster(rasters, outf, like=None, profile=None, tags=None):
    """Write a raster with the same profile as another raster.

    Parameters
    ----------
    rasters : np.array or similar; or list of np.array or similar
    outf : string
        output filename
    like : string, optional
        a filename of a raster whose profile should be used
    profile : dict, optional
        the profile dictionary from a raster object, if both like and profile
        are provided, values in profile will override values in like
    tags : dict or list of dicts, optional
        metadata to tag each band with

    """
    if type(rasters) is np.ndarray and len(rasters.shape) < 3:
        rasters = [rasters]

    if tags is None or type(tags) is dict:
        tags = [tags] * len(rasters)

    if len(tags) != len(rasters):
        raise ValueError(
            'An equal number of Rasters and Tags must be provided')

    if like is None and profile is None:
        raise ValueError('Either like or profile must be supplied')

    if outf.split('.')[-1] != 'tiff':
        raise ValueError('Only GeoTIFFs supported for now')

    if like is not None:
        with rio.open(like, 'r') as src:
            _profile = src.profile
    else:
        _profile = {}
    if profile is not None:
        _profile.update(profile)
    _profile['count'] = len(rasters)
    _profile['driver'] = 'GTiff'  # force for now

    warnstr = 'Writing array with dtype {} as raster with dtype {}'
    with rio.drivers():
        with rio.open(outf, 'w', **_profile) as dst:
            for i, raster in enumerate(rasters):
                bidx = i + 1
                raster[np.isnan(raster)] = _profile['nodata']
                if raster.dtype != np.dtype(_profile['dtype']):
                    warnings.warn(
                        warnstr.format(raster.dtype, _profile['dtype']))
                dst.write(raster.astype(_profile['dtype']), bidx)
                if tags[i] is not None:
                    dst.update_tags(bidx, **tags[i])


def combine_touched(shapes, like=None, profile=None, verbose=None):
    """Returns a raster with internal cell values set using `all_touched=False` and
    external cell values set with `all_touched=True`

    Parameters
    ----------
    shapes : multi-tuple or 2-tuple 
        tuples of geometry, values, as is fed to rasterio.features.rasterize
    all_touched : bool or int, optional
        if bool, interpreted as GDAL's rasterize option, if int > 1, a combined
        method is used taking border cells from all_touched=True and interior
        cells from all_touched=False
    like : string, optional
        a filename of a raster whose profile should be used
    profile : dict, optional
        the profile dictionary from a raster object
    verbose : bool, optional
        print out status information during rasterization

    """
    if like is None and profile is None:
        raise ValueError('Either like or profile must be supplied')

    if like is not None:
        with rio.open(like, 'r') as src:
            profile = src.profile

    shape = (profile['height'], profile['width'])
    transform = profile['affine']
    nodata = profile['nodata']
    dtype = profile['dtype']

    mask1 = rio.features.rasterize(shapes, all_touched=False,
                                   out_shape=shape,
                                   transform=transform, fill=nodata,
                                   dtype=dtype)
    if verbose:
        print('Done with mask 1')
    mask2 = rio.features.rasterize(shapes, all_touched=True,
                                   out_shape=shape,
                                   transform=transform, fill=nodata,
                                   dtype=dtype)
    if verbose:
        print('Done with mask 2')

    # add all border cells not covered by mask1 to mask1
    # Note:
    # must subtract nodata because we are adding to places where mask1 ==
    # nodata
    mask = mask1 + np.where((mask1 == nodata) & (mask2 != nodata),
                            mask2 - nodata, 0)
    return mask


def rasterize_indicies(shpf, all_touched=None, like=None, profile=None,
                       verbose=False, write=False):
    """Returns a raster with cell values set to the index of the corresponding
    shapefile feature.

    Parameters
    ----------
    shpf : string
        path to shapefile
    all_touched : bool or int, optional
        if bool, interpreted as GDAL's rasterize option, if int > 1, a combined
        method is used taking border cells from all_touched=True and interior
        cells from all_touched=False
    like : string, optional
        a filename of a raster whose profile should be used
    profile : dict, optional
        the profile dictionary from a raster object
    verbose : bool, optional
        print out status information during rasterization
    write : bool, optional
        if combined all_touched is used, all 3 masks are written to a file
        'masks.tiff' in the order: combined, all_touched=False, all_touched=True

    Returns
    -------
    raster : np.ndarray
        an array with cell values matched to the shapefile's features

    """
    if like is None and profile is None:
        raise ValueError('Either like or profile must be supplied')

    if like is not None:
        with rio.open(like, 'r') as src:
            profile = src.profile
    shape = (profile['height'], profile['width'])
    transform = profile['affine']
    _nodata = profile['nodata']
    dtype = profile['dtype']

    with fio.open(shpf, 'r') as n:
        geoms_idxs = tuple((c['geometry'], i) for i, c in enumerate(n))

    nodata = -1  # fails if some source raster nodatas are use
    all_touched = all_touched or 2  # default to combined strategy
    if all_touched < 2:
        mask = rio.features.rasterize(geoms_idxs,
                                      all_touched=all_touched, out_shape=shape,
                                      transform=transform, fill=nodata,
                                      dtype=dtype)
    else:
        mask1 = rio.features.rasterize(geoms_idxs,
                                       all_touched=False, out_shape=shape,
                                       transform=transform, fill=nodata,
                                       dtype=dtype)
        if verbose:
            print('Done with mask 1')
        mask2 = rio.features.rasterize(geoms_idxs,
                                       all_touched=True, out_shape=shape,
                                       transform=transform, fill=nodata,
                                       dtype=dtype)
        if verbose:
            print('Done with mask 2')
        # add all border cells not covered by mask1 to mask1
        # Note:
        # must subtract nodata because we are adding to places where mask1 ==
        # nodata
        mask = mask1 + np.where((mask1 == nodata) & (mask2 != nodata),
                                mask2 - nodata, 0)
        if verbose:
            print('Done with mask 3')
        if write:
            profile['nodata'] = nodata
            write_raster([mask, mask1, mask2], 'masks.tiff', profile=profile)

    mask[mask == nodata] = _nodata  # update to original nodata
    return mask
