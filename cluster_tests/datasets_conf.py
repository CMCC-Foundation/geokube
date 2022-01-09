ERA5 = "/data/inputs/ERA5/single-levels/reanalysis/*/*.nc"
ERA5_CHUNKS = {"latitude": -1, "longitude": -1, "time": 50}

ERA5_DOWNSCALED = "/work/remhi/mr29116/HIGHLANDER/v_newERA5/post_processing/pronti/*HLDRea_002_1hr_2001.nc"

E_OBS = "/data/inputs/E-OBS/mean/*_ens_*.nc"
E_OBS_PATTERN = (
    "/data/inputs/E-OBS/mean/{var}_ens_mean_{resolution}deg_reg_{version}.nc"
)
E_OBS_CHUNKS = ERA5_CHUNKS

NEMO_GLOBAL = "/data/products/OMIP/OMIP2-GLOB16/MONTHLY/ROMEO.01_1m_*_201510_grid_*.nc"
NEMO_GLOBAL_PATTERN = (
    "/data/products/OMIP/OMIP2-GLOB16/MONTHLY/ROMEO.01_1m_{}_{}_grid_{grid_type}.nc"
)
NEMO_GLOBAL_CHUNKS = {"x": -1, "y": -1, "time_counter": 1}
