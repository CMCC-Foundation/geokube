from typing import Mapping, Sequence, Self
import numpy.typing as npt
import pint

from .domain import Domain
from .field import Field
from .feature import Feature
from . import axis

# Cube is a set of multiple fields on the same domain!

class Cube():

    __slots__ = ('_dset', '__field_names', '__ancillary', '__coord_system')

    #
    # should we use compose here? use feature as an internal object
    # this will allow us to define only one cube
    # but we need to redefine the feature methods
    # 

    #TODO: improve the init to allow a mapping of multiple data with names
    # this implies to deal also with ancillary data, cell method and so on
    # maybe in this case we can allow only ancillary and cell method in common
    # to all fields? ...
    #
    
    def __init__(
        self,
        fields: Sequence[Field],
        domain: Domain | None = None # this should be used only in case we extend the init API as described in the TODO
    ) -> None:
        
        # TODO: check if all field are of the same type and defined on the same domain
        # 
        # TODO: check if fields have unique names
        #

        # merge fields in a unique xarray CF-compliant dataset
        # using the feature class of one of the field
        # 
        data_vars = {}
        self._field_names = []
        for f in fields:
            # field data
            self._field_names.append(f.name)
            data_vars[f.name] = f._dset[f.name]
            # ancillary data
            for anc, _ in f.ancillary.items():
                data_vars[anc] = f._dset[anc]
        
        # coords are built considering only 1 field since they are defined 
        # on the same domain
        coords = fields[0]._dset.coords

        feature_cls = fields[0].__class__
        self._dset = feature_cls(data_vars=data_vars,
                                 coords=coords,
                                 coord_system=fields[0].coord_system,
                                )

    #TODO: define __getitem__ as in xarray Dataset
    # key is an axis or a field name
    def __getitem__(self, 
                    key: axis.Axis | str) -> Field | dict[axis.Axis | pint.Quantity]:

        if isinstance(key, axis.Axis):
            # return the Coordinate data
            return self.coords[key]
        
        if isinstance(key, str):
            # return a field
            # check the ancillary data
            # 
            return self._dset[key]
    
    @property
    def coords(self) -> dict[axis.Axis, pint.Quantity]:
        coords = {}
        for ax in self.coord_system.axes:
            coords[ax] = self._dset[ax].pint.quantify().data
        return coords
    
    
    @property
    def coord_system(self) -> CoordinateSystem:
        return self.__coord_system