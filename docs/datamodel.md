# geokube Data Model
geokube Data Model defines the following data structure:
- `Axis` (key + type + properties) (xarray dimension/numpy axis)
- `Array` (data + axis + properties) - with axis operation are aware of axis properties (e.g. type)
- `Coordinate` (array + axis + bounds + properties) 
- `Domain` -> set of coordinates + crs + properties
- `Field` -> key + array + domain + properties
- `Cube`-> set of fields on a domain (defined as a merge of fields domains)
- `Frame` -> dataframe of cubes

### Axis
Axis is characterized by a key and properties.
The type property can belong to one of the following:
- X
- Y
- LATITUDE
- LONGITUDE
- VERTICAL
- TIME
- RADIAL_AZIMUTH
- RADIAL_DISTANCE
- RADIAL_ELEVATION
- GENERIC

Encoding contains information to be used when writing to a file format (netcdf, zarr, ...)

`Axis(key, properties, encoding)`

### Array
`Array` extends xarray Variable (a netcdf-like variable) consisting of dimnesions, data and properties.
Dimension in geokube Array is an Axis (key + AxisType)
Some operations are aware of axis type (e.g. resample, zonal_mean, ...) 
geokube Array take into accounts also units (represented as cf-units) in their operations 

`Array(data, dimensions, units, properties, encoding)`

### Coordinate
`Coordinate` can be of two types:

- `ArrayCoordinate` is an `Array` that is defined on an `Axis`, and optional `Bounds`. 
Two type of bounds
`Bounds1D` is a pandas `IntervalArray/IntervalIndex`. ArrayCoordinate represents data along an axis that could be an Array dimension or an auxiliary axis.
If Bounds are defined some operations should take that into account (regrid, resample, interpolation, ...)
`Bounds2D` ---> Array(axis+bnd)

- `ParametricCoordinate` is defined with a `ParametricFormula` (e.g python callable, lambda expression) expressed for each grid point identified by the dimensions. 
   Dict of `Array` representing values for the each term of the formula

<!-- (lat(rlat, rlon))

rlat -> Axis('rlat', 'Y')
rlon -> Axis('rlon', 'X')
lat -> Axis('lat', 'latitude')

latitude = Coordinate(Array([...], axis=rlat,rlon), lat)
rlat = Coordinate(Array([...], axis=rlat), rlat)

Coordinate([....], rlat)  ==> Coordinate(Array([...], axis=rlat), rlat)

Coordinate(([....], (rlat, rlon)), lat)  ==> Coordinate(Array([...], axis = [rlat,rlon]), lat)

Axis(x, 'generic')
Axis(nav_lat, 'latitude')

Z = Axis('z', Z)
height = Coordinate(Array(2.0), Z) -->

### Domain
`Domain` is a set of `Coordinates` and a CoordinateReferenceSystem for the Horizontal Grid

### Field
`Field` is an `Array` defined on a `Domain`

### Cube
`Cube` is a set of `Field` defined on a `Domain`

### Frame
`Frame` is a pandas dataframe of `Cube`. Columns are used as index for cubes.