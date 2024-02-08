import os
from tqdm import tqdm

import numpy as np
from pprint import pprint

import logging
logging.basicConfig(level=logging.INFO)

from ocean_tools.interfaces.swath.io import NetcdfFilesFTPSwotLRL2

#product_type = "basic"
#product_type = "wind-wave"
product_type = "expert"
#product_type = "unsmoothed"

orbit = "1day_orbit"

SWOT_LR_L2_ftp = f"/data/swot_beta_products/l2_karin/l2_lr_ssh/{orbit}/{product_type}"
#SWOT_LR_L2_ftp = f"/data/swot_beta_products/l2_karin/l2_lr_ssh/21day_orbit/{product_type}"

db = NetcdfFilesFTPSwotLRL2(SWOT_LR_L2_ftp)

print(db.last_date_available)

# 474, 577
all_cycles = list(range(474, 578))

cycles = all_cycles
#cycles = all_cycles[:10]
#passes = [1,2,3,4,5,6,7,8,9]
#passes = [3,] # med
#passes = [12,] # mascarene
passes = [7,20] # amazon
#passes = [21] # malucca
passes = sorted([3, 12, 7, 20, 21]) # all at once


#files = db.list_files(cycle_numbers=[523, 524], pass_numbers=[1,2,3,4,5,6,7,8,9], sort=True)
files = db.list_files(cycle_numbers=cycles, pass_numbers=passes, sort=True)
files = files["filename"].values.tolist()
print(f"number of files: {len(files)}")

# leverage common directories
ds = db.query(cycle_numbers=cycles, pass_numbers=passes)
print(ds)


print("All done (.py)")

