# gdal_utils
Wrappers for GDAL and OGR binaries and python bindings on Windows

## When to use this

1. Please use these central functions instead of your own wrappers to GDAL functions.
2. Please use [`rasterio`](https://mapbox.github.io/rasterio/) instead if you can. You can in any new systems you create that [do not `import gdal`](https://mapbox.github.io/rasterio/switch.html#mutual-incompatibilities).
