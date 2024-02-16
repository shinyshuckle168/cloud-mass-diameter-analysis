import os
from datetime import timedelta

import numpy as np
import xarray as xr, netCDF4, h5netcdf

def radar_merger(folder):
    # concatenates all netcdf files in folder along time dimension 
    radar_fnames = [f for f in os.listdir(folder) if f.endswith('.nc')]
    counter = 0
    for i, radar_fname in enumerate(radar_fnames):
        if counter == 0:
            merged_file = xr.open_dataset(os.path.join(folder, radar_fname)).isel(beam=slice(None,1)).isel(np=slice(None,2)).isel(nv=slice(None,2))
            counter = 1

        new_file = xr.open_dataset(os.path.join(folder, radar_fname)).isel(beam=slice(None,1)).isel(np=slice(None,2)).isel(nv=slice(None,2))
        merged_file = xr.concat([merged_file, new_file], dim='profile')
        print(f'Merged file {i}')

    return merged_file

if __name__ == '__main__':
    merged_file = radar_merger('D:/thesis/data/radar/f01_feb03/')
    merged_file.to_netcdf('D:/thesis/data/radar/f01_feb03/NAW_L1_merged.nc')