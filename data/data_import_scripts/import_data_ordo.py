import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import time
import sys
import urllib.request

# So that progress bar is shown rather than being blank during grab of file!
def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size),100)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

# Retrieve the file from URL link in ORDO
print('hody')
# urllib.request.urlretrieve("https://s3-eu-west-1.amazonaws.com/pstorage-ou-74982334129/14280608/openmars_my28_ls225_my28_ls245.nc", "data/my24part.nc", reporthook)
#urllib.request.urlretrieve("https://ordo.open.ac.uk/ndownloader/files/14280608", "my28part.nc", reporthook)
# This is one 30 sol file, of which in total we have 163 stored publicly! On 4MB/s speed downloaded in ~90 seconds
ds = xr.open_dataset('data/my24part.nc')
print(ds)
# Convert the dataset to a pandas dataframe
df = ds.to_dataframe()
print(df.sample(20))
# df.to_pickle('data/mars_4d_data')