from geokube import open_dataset, open_datacube
import xarray as xr
kube = open_dataset('data/prova.zarr', pattern='{MMM}/{yyyy:4}{mm:2}', use_zarr_groups_as_pattern=True, delay_read_cubes=True, root_group='data/prova.zarr', chunks={})

print(kube)

#kube = xr.open_dataset('data/climate-projections-zarr-test.zarr').QV_2M.attrs

#print(kube)