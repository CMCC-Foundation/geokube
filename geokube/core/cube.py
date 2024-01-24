from collections.abc import Sequence

import pint

from . import axis
from .coord_system import CoordinateSystem
from .domain import Domain
from .field import Field


# Cube is a set of multiple fields on the same domain!


class Cube:

    __slots__ = (
        '_dset', '_field_names', '_ancillary', '_coord_system', '_field_cls'
    )

    # TODO: Check if the slot `_ancillary` is needed.
    # TODO: Consider keeping the fields sequence.

    #
    # should we use compose here? use feature as an internal object
    # this will allow us to define only one cube
    # but we need to redefine the feature methods
    # 

    # TODO: improve the init to allow a mapping of multiple data with names
    # this implies to deal also with ancillary data, cell method and so on
    # maybe in this case we can allow only ancillary and cell method in common
    # to all fields? ...
    #

    def __init__(
        self,
        fields: Sequence[Field],
        domain: Domain | None = None  # this should be used only in case we extend the init API as described in the TODO
    ) -> None:

        # TODO: check if all field are of the same type and defined on the same domain
        # 
        # TODO: check if fields have unique names
        #

        # Checking whether types and domains are consistent, as well as whether
        # the names are unique.
        # TODO: Try to improve this part.
        types, domains, names = set(), [], set()
        for field in fields:
            types.add(type(field))
            domain_dset = Domain.as_xarray_dataset(
                coords={
                    axis_: coord.variable
                    for axis_, coord in field._dset.coords.items()
                },
                coord_system=field.coord_system
            )
            if not domains:
                domains.append(domain_dset)
            elif not domains[0].identical(domain_dset):
                raise ValueError(
                    "'fields' must be a collection of 'Field' objects with the "
                    "same domain"
                )
            field_name = field.name
            if field_name in names:
                raise ValueError(
                    "'fields' must be a collection of 'Field' objects with "
                    "unique names"
                )
            names.add(field_name)
        if len(types) > 1:
            raise TypeError(
                "'fields' must be a collection of 'Field' objects of the "
                "same type"
            )

        # merge fields in a unique xarray CF-compliant dataset
        # using the feature class of one of the field
        # 
        data_vars = {}
        field_names = []
        for f in fields:
            # field data
            field_names.append(f.name)
            data_vars[f.name] = f._dset[f.name]
            # ancillary data
            for anc in f.ancillary:
                data_vars[anc] = f._dset[anc]

        # coords are built considering only 1 field since they are defined 
        # on the same domain
        # coords = dict(fields[0]._dset.coords)
        # coord_system = fields[0].coord_system
        dset = Domain.as_xarray_dataset(
            coords={
                axis_: coord.variable
                for axis_, coord in fields[0]._dset.coords.items()
            },
            coord_system=fields[0].coord_system
        )
        dset = dset.drop_indexes(coord_names=list(dset.xindexes.keys()))
        dset = dset.assign(variables=data_vars)

        field_cls = fields[0].__class__
        self._field_cls = field_cls
        # TODO: Make sure that `field_cls.__mro__[2]` returns the correct
        # `Feature` class from the inheritance hierarchy.
        feature_cls = field_cls.__mro__[2]
        feature = feature_cls(dset)
        self._dset = feature._dset
        self._coord_system = feature.coord_system
        self._field_names = tuple(field_names)
        # self._dset = feature_cls(data_vars=data_vars,
        #                          coords=coords,
        #                          coord_system=coord_system,
        #                         )

    # TODO: define __getitem__ as in xarray Dataset
    # key is an axis or a field name
    def __getitem__(self, key: axis.Axis | str) -> Field | pint.Quantity:

        # TODO: Clear this logic with `str` keys.
        # TODO: Check the return type.

        if isinstance(key, axis.Axis):
            # return the Coordinate data
            return self._dset.coords[key].pint.quantify().data

        if isinstance(key, str):
            # return a field
            # check the ancillary data
            # 
            # TODO: Include ancillary vars into `needed_vars`.
            needed_vars = {key}
            redundant_vars = self._dset.data_vars.keys() - needed_vars
            dset = self._dset.drop_vars(names=redundant_vars)
            dset = dset.drop_indexes(coord_names=list(dset.xindexes.keys()))
            # dset.attrs[_FIELD_NAME_ATTR_] = key
            return self._field_cls._from_xarray_dataset(dset)

        raise TypeError("'key' must be an instance of 'str' or 'Axis'")

    @property
    def coords(self) -> dict[axis.Axis, pint.Quantity]:
        return {
            ax: self._dset[ax].pint.quantify().data
            for ax in self.coord_system.axes
        }

    @property
    def coord_system(self) -> CoordinateSystem:
        return self._coord_system
