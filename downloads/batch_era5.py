
import os
import cdsapi
import pandas as pd

assert "http_proxy" in os.environ
assert "https_proxy" in os.environ

overwrite=False

#OUTPUT_DIR = f'/work/scratch/data/{os.environ["USER"]}/tmp_data'
OUTPUT_DIR = f"/work/HELPDESK_SWOTLR/swot_diego/era5"

cds_client = cdsapi.Client()

#dates = pd.date_range("2023/03/01", "2023/07/01", freq="1D")
dates = pd.date_range("2023/07/01", "2023/10/01", freq="1D")

for day in dates:

    output_file = os.path.join(OUTPUT_DIR,f'reanalysis-era5-single-levels_{day.strftime("%Y%m%d")}.nc')

    if not os.path.isfile(output_file) or overwrite:
        cds_client.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '10m_u_component_of_wind', '10m_v_component_of_wind', 'instantaneous_eastward_turbulent_surface_stress',
                    'instantaneous_northward_turbulent_surface_stress', 'significant_height_of_combined_wind_waves_and_swell', 'significant_height_of_wind_waves',
                    'surface_pressure',
                ],
                'year': day.year,
                'month': day.month,
                'day': day.day,
                'time': ['%02d:00'%(hour) for hour in range(24)],
            },
            output_file)


