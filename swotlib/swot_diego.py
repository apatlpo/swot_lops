import os, sys
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
import datetime

import matplotlib.pyplot as plt
import cartopy
crs = cartopy.crs.PlateCarree()
import cartopy.feature as cfeature
import pyproj

if os.path.isdir("/home/datawork-lops-osi/"):
    platform="ifremer"
else:
    platform="cnes"


# CLS libraries
import pyinterp

if platform=="ifremer":
    pass
elif platform=="cnes":
    import swot_calval.io
    from assumerole import assumerole
    import s3fs

    # load external libraries
    
    root_path = "/work/HELPDESK_SWOTLR/swot_diego/"
    
    lib_path = os.path.join(root_path, 'libs')
    sys.path.append(os.path.join(lib_path, 'SWOT-OpenToolkit/src/'))
    #sys.path.append(os.path.join(lib_path, 'SWOT-OpenToolkit/data/'))
    sys.path.append(os.path.join(lib_path, 'search_swot-master'))
    
    fig_dir = f'/work/scratch/data/{os.environ["USER"]}/figs/'
    

# ---------------------- common --------------------------------

def get_passes_times(coordinates):

    SWOT_DATA = '/work/HELPDESK_SWOTLR/commun/data/swot/L3_LR_SSH/v0.2/zcoll'
    collection = swot_calval.io.open_collection(SWOT_DATA)

    LATITUDE = 'latitude'
    LONGITUDE = 'longitude'
    TIME = 'time'
    selection = [TIME, LONGITUDE, LATITUDE,]
    
    polygon = pyinterp.geodetic.Polygon.from_geojson(coordinates)
    
    half_orbits = collection.query_half_orbits(
        first_day=datetime.date(2023, 4, 1), last_day=datetime.date(2023, 7, 15),
        selected_variables=[
            'cycle_number',
            'pass_number',
        ]+selection,
        polygon=polygon,
    )

    D = []
    for (cycle, pass_number), zds in half_orbits.items():
        D.append(dict(cycle=cycle, 
                      pass_number=pass_number, 
                      time=zds.to_xarray().time.mean().values
                     )
                )
    return pd.DataFrame(D)


def spin_up_cluster(jobs=1, **kwargs):
    
    import dask_jobqueue
    from distributed import Client

    dkwargs = dict(
        cores=1,
        memory='8GiB', # Balanced configuration should have 8 GiB per core if possible
        processes=1,
        log_directory=f'/work/scratch/data/{os.environ["USER"]}/dask_stuff',
        walltime='01:00:00',
        interface='ib0',
        # Please fill your account, obtained  with the command $myaccounts
        account='swotce_guest',
        job_extra_directives=['--export=None'], # Do not propagate environment
    )
    dkwargs.update(**kwargs)
    cluster = dask_jobqueue.SLURMCluster(**dkwargs)
    cluster.scale(jobs=jobs)
    client = Client(cluster)

    #client.wait_for_workers(1)
    return client, cluster
    
# plotting
    
def clean_figdir():
    files = glob(os.path.join(fig_dir, '*'))
    for f in files:
        os.remove(f)

def thumbnails(
    half_orbits, v, plotter,
    ny=1, nx=5, figsize=(15,8),
    suffix=None, dry_run=False,
    **kwargs,
    ):
    ''' plot thumbnails '''
    
    i, ii = 0, 0
    for (cycle, pass_number), zds in half_orbits.items():

        if i%(nx*ny)==0:
            # print&close figure
            if suffix is not None and not dry_run and i>0:
                fig.savefig(os.path.join(fig_dir, f'thumbnails_{suffix}_{ii:03d}.png'))
                plt.close(fig)
            fig = plt.figure(figsize=figsize) #layout='constrained', 
            gs = fig.add_gridspec(ny, nx, hspace=0, wspace=0)
            i=0
            ii+=1

        j, k = np.unravel_index(i, (ny, nx))
        ax = fig.add_subplot(gs[j, k])
        plotter(zds, v, cycle, pass_number, ax, **kwargs)

        i+=1        
        if ii>=1 and i%(nx*ny)==0 and dry_run:
            break
            
def thumbnails_with_wind(
    half_orbits, v, plotter, era5,
    nx=5, figsize=(15,8),
    suffix=None, dry_run=False,
    **kwargs,
    ):
    ''' plot thumbnails '''
    
    i, ii = 0, 0
    while i<len(half_orbits):

        i_range = list(range(i,i+nx))

        fig = plt.figure(figsize=figsize) #layout='constrained', 
        gs = fig.add_gridspec(4, nx, hspace=.2, wspace=0)
    
        # map plot
        j, k = 0, 0
        D = []
        for (cycle, pass_number), zds in half_orbits.items():
            if j in i_range:
                ax = fig.add_subplot(gs[:-1, k])
                start, end = plotter(zds, v, cycle, pass_number, ax, **kwargs)
                D.append(dict(cycle=cycle,
                              pass_number=pass_number, 
                              start=start,
                              end=end,
                             ))
                k+=1
            j+=1
        df = pd.DataFrame(D)

        # wind
        ax = fig.add_subplot(gs[-1, :])
        show_wind_teries(era5, ax=ax) # label=region,
        for idx, r in df.iterrows():
            try:
                if ~pd.isnull(r['start']):
                    ax.axvline(r['start'], color='k')
            except:
                print(r['start'], type(r['start']))
        # reset x limit
        dt = pd.Timedelta('5D')
        ax.set_xlim(df['start'].min()-dt, df['end'].max()+dt)
        ax.set_ylim(-20,20)
        
        # print&close figure
        if suffix is not None and not dry_run:
            fig.savefig(os.path.join(fig_dir, f'thumbnails_wd_{suffix}_{ii:03d}.png'))
            plt.close(fig)
        ii+=1
        
        if ii>=1 and dry_run:
            break

        i+=nx
        
def thumbnail_multi_variables(
    variables, plotter,
    zds, cycle, pass_number, i,
    era5=None, wind_lims=(-20,20),
    tides=None, tides_lims=(-1,1),
    dt='10D',
    figsize=(10,10), ny=4,
    suffix=None, dry_run=False, overwrite=False,
):

    fig_path = os.path.join(fig_dir, f'thumbnails_multiv_{suffix}_{i:03d}.png')
    if not overwrite and os.path.isfile(fig_path) and not dry_run:
        return
    
    nx = len(variables)

    fig = plt.figure(figsize=figsize) #layout='constrained', 
    gs = fig.add_gridspec(ny, nx, hspace=0.1, wspace=0)

    j, starts, ends = 0, [], []
    for d in variables:
        kwargs = dict(**d)
        del kwargs['label']
        ax = fig.add_subplot(gs[:-1, j])
        start, end = plotter(zds, d['label'], cycle, pass_number, ax, **kwargs)
        starts.append(start)
        ends.append(end)
        j+=1

    # exit if no data
    if all([pd.isnull(s) for s in starts]):
        plt.close(fig)
        return
    start = [s for s in starts if not pd.isnull(s)][0]
    end = [e for e in ends if not pd.isnull(e)][0]
        
    # wind
    dt = pd.Timedelta(dt)
    if tides is None:
        ax = fig.add_subplot(gs[-1, :])
    else:
        ax = fig.add_subplot(gs[-1, :ny//2])
    show_wind_teries(era5, ax=ax) # label=region,
    ax.axvline(start, color='k', lw=2)
    ax.set_xlim(start-dt, end+dt)
    ax.set_ylim(*wind_lims)

    if tides is not None:
        ax = fig.add_subplot(gs[-1, ny//2:])
        tides.plot(ax=ax, color="0.5", lw=2) # label=region,
        ax.axvline(start, color='k', lw=2)
        ax.set_xlim(start-dt, end+dt)
        ax.set_ylim(*tides_lims)
        ax.yaxis.tick_right()
        ax.grid()
        
    if dry_run or suffix is None:
        return
    fig.savefig(fig_path)
    plt.close(fig)

def add_geo_projections(ds, lonc=None, latc=None, phi=None):
    
    # L2 case
    if isinstance(ds, dict):
        kwargs = dict(lonc=lonc, latc=latc, phi=phi)
        return {side: add_geo_projections(ds[side], **kwargs) for side in ['left', 'right']}

    i = ds.num_pixels.size//2
    j = ds.num_lines.size//2

    if lonc is None or latc is None:
        lonc = float(ds.isel(num_pixels=i, num_lines=j).longitude.values)
        latc = float(ds.isel(num_pixels=i, num_lines=j).latitude.values)
    proj = _get_projection(lonc, latc)
        
    if phi is None:
        lon0, lon1 = ds.isel(num_pixels=i, num_lines=[0,-1]).longitude.values
        lat0, lat1 = ds.isel(num_pixels=i, num_lines=[0,-1]).latitude.values
        phi = _compute_orientation(lonc, latc, lon0, lat0, lon1, lat1) # in degrees
    
    x, y = _lonlat2xy(lonc, latc, phi, ds.longitude, ds.latitude)
    ds['x'], ds['y'] = (ds.longitude.dims, x/1e3), (ds.longitude.dims, y/1e3)
    ds['x'], ds['y'] = fillna_plane(ds) # fill x/y NaN
    ds['x'].attrs['units'] = 'km'
    ds['y'].attrs['units'] = 'km'
    ds = ds.set_coords(['x', 'y'])

    # store useful information about projections
    ds.attrs['proj_phi'] = phi
    ds.attrs['proj_lonc'] = lonc
    ds.attrs['proj_latc'] = latc    
    
    return ds
    
def _get_projection(lonc, latc):#converts from lon, lat to native map projection x,y
    ''' Create pyproj Proj object, project is an azimutal Eqsuidistant projection centered on the central point of the selected satellite track = matching point
    https://proj.org/operations/projections/aeqd.html
            
    Parameters
    ----------
    lonc,latc : float
        central longitude and latitude of the satellite track, matching point on which the box will be centered
    Return
    ------
    pyproj.Proj object
    '''
    return pyproj.Proj(proj='aeqd', lat_0=latc, lon_0=lonc, datum='WGS84', units='m') #aeqd Azimutal EQuiDistant projection centered on lonc,latc

def _compute_orientation(lonc, latc, lon0, lat0, lon1, lat1):
    '''Compute the orientation of the box i.e. the angle (-180 and 180°) between the oriented track (time=0 -> end) and the longitude axe.
        
    Parameters
    ----------
    lonc,latc : float
        central longitude and latitude of the satellite track, matching point on which the box will be centered
    lon1,lat1 : float
        longitude and latitude of the end of the satellite track
    Return
    ------
    box orientation
    '''
    proj = _get_projection(lonc, latc)
    # get local coordinate
    x0, y0 = proj.transform(lon0, lat0) #xc=yc=0 origin of the box grid
    x1, y1 = proj.transform(lon1, lat1)
    # get orientation of defined by central point and point 1
    
    phi = np.arctan2( y1-y0, x1-x0 )*180/np.pi # angle (-180 and 180°) between the oriented track (time=0 -> end) and the longitude axe
    #assert  phi==0, (x0,y0, x1,y1)
    return phi

def _lonlat2xy(lonc, latc, phi, lon, lat):
    ''' return coordinates with origin at (lonc, latc) and x-axis aligned 
    with (lonc, latc) - (lon1, lat1) direction (lon,lat -> x-along satellite track, y-normal to satellite track)
    
    Parameters
    ----------
    lonc, latc, phi: float  
        central position and orientation of the box
    lon, lat : np.array, np.array     
        local grid of the box
    lon1,lat1 : float      
        end of the satellite track
    
    Return
    ------
    local x, local y : np.array, np.array
      
    '''
    proj = _get_projection(lonc, latc)
    # get local coordinate
    xc, yc = proj.transform(lonc, latc)    
    xl, yl = proj.transform(lon, lat)
    # build affine operators

    # requires rasterio.Affine
    #a_fwrd = Affine.translation(-xc, -yc) * Affine.rotation(-phi, pivot=(xc, yc))
    ##a_back = ~a_fwrd
    #x, y = a_fwrd * (xl, yl)
    x, y = _rotate(xl, yl, np.pi/2-np.deg2rad(phi))
    
    return x, y

def _rotate(x, y, phi):
    ''' rotate coordinates, phi must be in radians '''
    cos, sin = np.cos(phi), np.sin(phi)
    return x*cos - y*sin, x*sin + y*cos


def fillna_plane(da):
    ''' fill x/y coordinate NaN values with plane fit '''

    if isinstance(da, xr.Dataset):
        return fillna_plane(da['x']), fillna_plane(da['y'])
    
    #fit = da.polyfit(['num_lines', 'num_pixels'], 1)
    #da_out = xr.polyval(coord=da.num_lines, coeffs=fit['polyfit_coefficients'])

    func = lambda x, p0, p_lines, p_pixels: p0+p_lines*x[0]+p_pixels*x[1]
    coords = ['num_lines', 'num_pixels']
    fit = da.curvefit(coords, func)
    da_fit = func([da[c] for c in coords], *fit['curvefit_coefficients'])
    da_out = da.fillna(da_fit)
    
    return da_out

def interp_da(da, ds_target, inplace=True, **kwargs):
    ''' interpolate dataarray on a different dataset'''

    mesh = pyinterp.RTree()
    lon = da['longitude'].values
    lat = da['latitude'].values
    v = da.values
    mesh.packing(np.vstack((lon.ravel(), lat.ravel())).T, v.ravel())

    lon_target, lat_target = ds_target['longitude'].values, ds_target['latitude'].values
    
    dkwargs = dict(
        within=False,  # Extrapolation is forbidden
        radius=10_000,  # In a radius of 10 Km
        k=8,  # We are looking for at most 8 neighbours
        num_threads=0,
    )
    dkwargs.update(**kwargs)
    
    idw_v, neighbors = mesh.inverse_distance_weighting(
        np.vstack((lon_target.ravel(), lat_target.ravel())).T,
    )
    idw_v = idw_v.reshape(lon_target.shape)
    
    if inplace:
        ds_target[da.name] = (da.dims, idw_v) 
    else:
        return xr.DataArray(idw_v, coords=ds_target.coords, name=da.name)


def rescale_sigma0(ds, mss=2.5e-2):
    ''' 
    rescale sigma0 to limit cross-track sensitivity
    add tan(theta) where theta is the incidence angle
    inplace additions
    mss = 2.5e-2 is an average value, for u10 of about 5 m/s
    '''
    add_tan_theta(ds)
    if 'sig0_karin_2' in ds:
        ds['sig0_karin_2_raw'] = ds['sig0_karin_2']
        ds['sig0_karin_2'] =  ds['sig0_karin_2'] * np.exp(ds.tan_theta**2/mss)
        # note: overwrites orignal variable (easier for subsequent preprocessing)

def _rescale_sigma_pierre(ds):
    a = 6378137
    beta = np.abs(ds.cross_track_distance) / a
    if "sc_altitude" in ds:
        altitude = ds.sc_altitude
    else:
        altitude = 891e3
    r = np.sqrt(6378137.**2 + (altitude + a)**2 - 2 * np.cos(beta) * a*(altitude + a) )
    theta = np.arccos((r**2 + (altitude + a)**2 - a**2) / (2 * r*(altitude + a) ))
    incs = theta + beta
    def sig0Nadir(ds,i=0,j=0):
        R2 = 0.56
        coeff = [
            ds.sig0_karin[i,j]/R2 * np.cos(incs[i,j]),
            -1,
            np.tan(incs[i,j])**2,
            -np.tan(incs[i,j])**4/2,
        ] # cubic function
        xx = np.roots(coeff)
        mss = xx[xx.imag==0][0].real
        return R2/mss
    sig0Nadir(ds)
        
def add_tan_theta(ds):
    """ tangent of incidence angle, inplace addition """
    if "cross_track_distance" in ds:
        R=891e3 # m satellite altitude
        ds['tan_theta'] = ds.cross_track_distance/891e3
    elif "num_pixels" in ds.dims:
        ds['tan_theta'] = (
            'num_pixels', 
            np.tan(np.deg2rad(np.linspace(0.6, 4.1, ds.num_pixels.size)))
        )
        # from 0.01 to 0.07
    else:
        assert False, 'cannot estimate tan_theta'
        
def zoom(ds, extent):
    ''' geographical zoom '''
    
    # compute these variables or crash later
    ds = ds.copy()
    for v in ["longitude", "latitude", "time"]:
        ds[v] = ds[v].compute()
    ds["time"], _ = xr.broadcast(ds.time, ds.longitude)
    
    #ds1 = (
    #    ds
    #    .isel(num_pixels=0)
    #    .reset_coords()
    #    [["longitude", "latitude"]]
    #)
    ds1 = (
        ds.reset_coords()[["longitude", "latitude"]]
        .median('num_pixels')
    )
    ds1 = ds1.assign_coords(num_lines=ds1.num_lines)
    lon, lat = ds1.longitude, ds1.latitude
    mask = (
        (lon > extent[0]) 
        & (lon < extent[1])
        & (lat > extent[2])
        & (lat < extent[3])
    )    
    ds1 = ds1.where(mask, drop=True)
    
    if ds1.num_lines.size>0:
        #return ds.isel(num_lines=ds1.num_lines) # bug ?
        return ds.isel(num_lines=slice(int(ds1.num_lines[0]), int(ds1.num_lines[-1])))
    else:
        return None
    
    # bug ?
    #coords = list(ds.coords)
    #try:
    #    ds = (
    #        ds
    #        .reset_coords()
    #        .where(mask, drop=True)
    #        .set_coords(coords)
    #    )
    #except:
    #    # no points in this area
    #    ds = None
    #
    #return ds
    

def mask_filter(
    da, 
    mask_level=1,
    spike_threshold=None,
    spike_atrack=3, spike_xtrack=3,
    spike_normalize=False,
    fillna = False,
    lbd=10e3, dx=250, truncate=2,
    dev=False,
):
    """ 
    1/ Mask data with provided mask
    2/ Filter out spikes with 2D rolling median (3x3) and  - 2 passes. 
       Applied on raw or normalized differences: i.e. v-v_smoothed (default) or (v-v_smoothed)/v_smoothed
       Applied twice
    3/ Fill NaN - optional, default is not to
    3/ Decompose into low/high components via Gaussian filtering
    
    Parameters
    ----------
    da: xr.DataArray
        Array to be smoothed
    mask_level: int, optional
        Level used to mask based on mask provided alongside data
    spike_threshold: float
        Threshold employed for spike detection
    spike_atrack, spike_xtrack: int
        window sizes used for rolling median and despiking
    spike_normalize: boolean
        whether to normalize or not differences while despiking
    fillna: boolean
        turns on NaN filling prior to spatial smoothing
    lbd: float
        size of the gaussian kernel
    dx: float
        grid size truncate = 2,    # gaussian filtering
    dev: boolean
        outputs complete set of arrays for development    
    """
    from scipy.ndimage import gaussian_filter

    # put everything in memory, required for some steps
    da = da.compute()
    mask = da.mask.compute()

    ## mask
    ds = da.rename("start").to_dataset()
    if mask_level is not None:
        da = da.where(mask<mask_level)
    ds["masked0"] = da

    ## small scale spike detection, applied twice
    
    # pass 0
    ds["smoothed0"] = da.rolling(dict(num_lines=spike_atrack, num_pixels=spike_xtrack), center=True).median()
    if spike_normalize:
        _mask = (np.abs(da - ds.smoothed0)<spike_threshold)
    else:
        _mask = (np.abs((da - ds.smoothed0)/ds.smoothed0)<spike_threshold)
    da = da.where( _mask & (~np.isnan(ds.smoothed0)) )
    ds["masked1"] = da

    # pass 1
    ds["smoothed1"] = da.rolling(dict(num_lines=spike_atrack, num_pixels=spike_xtrack), center=True).median()
    if spike_normalize:
        _mask = (np.abs(da - ds.smoothed1)<spike_threshold)
    else:
        _mask = (np.abs((da - ds.smoothed1)/ds.smoothed1)<spike_threshold)
    da = da.where( _mask & (~np.isnan(ds.smoothed1)) )
    ds["masked2"] = da

    ## fill NaN - optional, default is not to
    if fillna:
        da = (
            da
            .interpolate_na("num_pixels", method="linear")
            .interpolate_na("num_lines", method="linear")
            .interpolate_na("num_pixels", method="nearest", fill_value="extrapolate")
            .interpolate_na("num_lines", method="nearest", fill_value="extrapolate")
        )
    ds["filled"] = da

    ## Gaussian filtering
    sigma = lbd/dx
    kw = dict(sigma=(sigma,)*2, truncate=truncate)
    _da = gaussian_filter(da.fillna(0.), **kw)
    _w = gaussian_filter((0*da+1).fillna(0.), **kw)
    ds["smoothed2"] = (da.dims, _da/_w)

    da_low = ds.smoothed2
    da_high = da - ds.smoothed2

    if dev:
        return da_low, da_high, ds

    return da_low, da_high

def plot_map(extent, figsize=(15,10), fig=None, ax=None):

    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(1, 1, 1, projection=crs)
    ax.set_extent(extent, crs=crs)

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, zorder=5)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    #ax.set_title(ds.time.dt.strftime('%Y/%d/%m %H:%M').values[0])

    return fig, ax


def load_bathy(info):
    """ load bathymetry 
    Bathymetry must be downloaded from GEBCO: https://download.gebco.net
    Path to netcdf file must be specified in info["bathy"]
    """
    ds = xr.open_dataset(info["bathy"])
    ds["depth"] = -ds.elevation.where(ds.elevation<0)
    return ds
    
# ---------------------- L2 --------------------------------

# ssh_karin_2: Fully corrected sea surface height measured by KaRIn. The height is relative to the reference ellipsoid defined in the global attributes. This value is computed using model-based estimates for wet troposphere effects on the KaRIn measurement (e.g., model_wet_tropo_cor and sea_state_bias_cor_2).

# ssha_karin_2 = ssh_karin_2 - mean_sea_surface_cnescls - solid_earth_tide - ocean_tide_fes – internal_tide_hret - pole_tide - dac.

# dict to access variable names easily
l2 = dict(
    core=['time', 'longitude', 'latitude', 'cross_track_distance'],
    ssh=[
        'ssh_karin_2', 'ssh_karin_2_qual', 'ssh_karin_uncert',
        'mean_sea_surface_cnescls', 'cvl_mean_sea_surface_cnes_22_hybrid', 'cvl_geoid',
        'cvl_mean_dynamic_topography_cnes_cls_22',
        'cvl_height_cor_xover', # xcal in L3
        'cvl_solid_earth_tide', 'cvl_ocean_tide_fes_2022', 'cvl_internal_tide_hret', 'cvl_pole_tide',
        'cvl_dac',
        'cvl_height_cor_l3', 'cvl_phase_screen_l3',
        #'sea_state_bias_cor_2', 'cvl_sea_state_bias_cor_model', # sea_state_bias_cor_2: Sea state bias correction used to compute ssh_karin_2. Adding the reported correction to the reported sea surface height results in the uncorrected sea surface height. The wind_speed_karin_2 value is used to compute this quantity.  
        'cvl_model_dry_tropo_cor', 'cvl_model_wet_tropo_cor', 
    ],
    sig0=[
        'sig0_karin_2', 'sig0_karin_2_qual', 'sig0_karin_uncert',
    ],
)

def load_l2(
    #orbit='1d',
    variables=None,
    geoselection=None,
    half_orbits=True,
    probe_variables=False,
    return_collection=False,
    **kwargs
):
    ''' load L2 data via zcoll
    
    To inspect variables: load_l2(probe_variables=True)
    
    Parameters
    ----------
    variables: list, optional
        load all variables by default
    geoselection: list of list
        geographical selection
    half_orbits: boolean
        return half orbits instead of concatenated fields
    probe_variables: boolean
        print variables names
    '''

    _kwargs = dict()    

    # open collection
    credentials = {"AWS_ACCESS_KEY_ID":"", "AWS_SECRET_ACCESS_KEY":"", "AWS_SESSION_TOKEN":""}
    credentials = assumerole.getCredentials('arn:aws:iam::888937369709:role/swotce-zcollection-public-role')
    # credentials = assumerole.getCredentials('arn:aws:iam::888937369709:role/swotce-readonly')    
    fs = s3fs.S3FileSystem(
        key=credentials['AWS_ACCESS_KEY_ID'],
        secret=credentials['AWS_SECRET_ACCESS_KEY'],
        token=credentials['AWS_SESSION_TOKEN'],
        config_kwargs={'connect_timeout': 240, 'read_timeout': 240},
        client_kwargs=dict(
            region_name='us-east-1',
            endpoint_url='https://s3.datalake.sis.cnes.fr',
            verify=False),
    )    
    collection = swot_calval.io.open_collection(
        'swotce-zcollection/zcollection_lr_ocean_unsmoothed',
        mode='r',
        filesystem=fs,
    )
    if return_collection:
        return collection
    
    if probe_variables:
        for label, v in collection.variables().items():
            print('--- '+label)
            if len(v.attrs)>0:
                for a in v.attrs:
                    print(' ', a.name, ': ', a.value)            
        #print('\n'.join(list(collection.variables())))
        return
    
    if variables is not None:
        variables = list(set(variables + l2['core'])) # removes duplicates

    if geoselection is not None:
        polygon = pyinterp.geodetic.Polygon.from_geojson(geoselection)
        _kwargs['polygon'] = polygon

    if half_orbits:
        query = collection.query_half_orbits
        if variables is not None:
            variables += ['cycle_number', 'pass_number']
    else:
        query = collection.query

    _kwargs = dict(selected_variables=variables)
    _kwargs.update(**kwargs)

    return query(**_kwargs)


SWOT_L2_CACHE = '/work/HELPDESK_SWOTLR/commun/data/swot/cache/'
def load_l2_std(product_type, pass_number, cycle_number, extent=None, preprocessing=None, s0_rescale=True):
    ''' load one l2 pass cycle
    '''
    
    files = sorted(
        glob(
            os.path.join(
                SWOT_L2_CACHE, 
                f'SWOT_L2_LR_SSH_{product_type}_{cycle_number}_{pass_number:03d}*.nc')
            )
    )
    if len(files)!=1:
        return None
    #assert len(files)==1, f'more than one file matching:'+' , '.join([f for f in files])
    file = files[0]
    
    left = xr.open_dataset(file, group='left')
    right = xr.open_dataset(file, group='right')
    #add_incidence(left)
    #add_incidence(right)
    if s0_rescale:
        rescale_sigma0(left)
        rescale_sigma0(right)
    
    if extent is not None:
        left = zoom(left, extent)
        right = zoom(right, extent)
    if left is None or right is None:
        return None

    if preprocessing is not None:
        left = preprocessing(left)
        right = preprocessing(right)
        
    return dict(left=left, right=right)


# ---------------------- L3 --------------------------------


# see Tutorials / Explore SWOT data

# ssha_karin_2 = ssh_karin_2 - mean_sea_surface_cnescls - solid_earth_tide - ocean_tide_fes – internal_tide_hret - pole_tide - dac.

# dict to access order variables easily
l3 = dict(
    core=['time', 'longitude', 'latitude', 'cross_track_distance'],
    ssh=[
        'ssha_karin_2_filtered', 'ssha_karin_2', 'ssha_karin_2_qual', 'ssh_karin_uncert', 'ssha_karin_2_calibrated', 
        'mean_sea_surface_cnescls', 'cvl_mean_dynamic_topography_cnes_cls_22', 'cvl_mean_sea_surface_cnes_22_hybrid',
        'xcal', # cvl_height_cor_xover in L2
        #'cvl_solid_earth_tide', 'cvl_pole_tide',
        'cvl_ocean_tide_fes_2022', 'internal_tide_hret', 
        'dac', # cvl_dac in L2
        'sea_state_bias_cor_2', 'cvl_sea_state_bias_cor_model', # sea_state_bias_cor_2: Sea state bias correction used to compute ssh_karin_2. Adding the reported correction to the reported sea surface height results in the uncorrected sea surface height. The wind_speed_karin_2 value is used to compute this quantity.  
    ],
    ssh_dyn=[
        'ssha_karin_2_filtered', 'ssha_karin_2', 'ssha_karin_2_qual', 'ssh_karin_uncert', #'ssha_karin_2_calibrated',
        'cvl_mean_dynamic_topography_cnes_cls_22',
    ],
    sig0=[
        'sig0_karin_2', # sig0_karin_2_qual (not available)
    ],
    speed=[
        'speed_meridional', 'speed_meridional_abs',
        'speed_zonal', 'speed_zonal_abs', 
        'strain', 'relative_vorticity',
        
    ]
)

def load_l3(
    orbit='1d',
    variables=None,
    geoselection=None,
    half_orbits=True,
    probe_variables=False,
    return_collection=False,
    **kwargs
):
    ''' load L3 data via zcoll
    
    To inspect variables: load_l3(probe_variables=True)
    
    Parameters
    ----------
    orbit: str
        '1d' (default) or '20d'
    variables: list, optional
        load all variables by default
    geoselection: list of list
        geographical selection
    half_orbits: boolean
        return half orbits instead of concatenated fields
    probe_variables: boolean
        print variables names
    '''

    _kwargs = dict()    

    if orbit=='1d':
        SWOT_DATA = '/work/HELPDESK_SWOTLR/commun/data/swot/L3_LR_SSH/v0.2/zcoll'
    elif orbit=='20d':
        SWOT_DATA = '/work/HELPDESK_SWOTLR/commun/data/swot/L3_LR_SSH/v0.2_science/zcoll'
    collection = swot_calval.io.open_collection(SWOT_DATA)
    if return_collection:
        return collection

    if probe_variables:
        for label, v in collection.variables().items():
            print('--- '+label)
            if len(v.attrs)>0:
                for a in v.attrs:
                    print(' ', a.name, ': ', a.value)            
        #print('\n'.join(list(collection.variables())))
        return
    
    if variables is not None:
        variables = l3['core'] + variables

    if geoselection is not None:
        polygon = pyinterp.geodetic.Polygon.from_geojson(geoselection)
        _kwargs['polygon'] = polygon

    if half_orbits:
        query = collection.query_half_orbits
        if variables is not None:
            variables += ['cycle_number', 'pass_number']
    else:
        query = collection.query

    _kwargs = dict(selected_variables=variables)
    _kwargs.update(**kwargs)

    return query(**_kwargs)


def plot_l3_map(
    cycle, pass_number, 
    ds,
    extent,
    v, vkwargs={},
    a=None, a_di=1, a_ref=1, akwargs={},
    close=False,
    fig=None,
    ax=None,
):

    if fig is None:
        fig = plt.figure(figsize=(15, 10))
    if ax is None:
        ax = fig.add_subplot(1, 1, 1, projection=crs)
    ax.set_extent(extent, crs=crs)

    #(ds['ssha'] + ds['mdt']).rename('ssh')
    if isinstance(v, tuple):
        da = sum([ds[_v] for _v in v])
    elif isinstance(v, str):
        da = ds[v]
    
    kwargs = dict(
        x='longitude',
        y='latitude',
        ax=ax,
        transform=crs,
        extend='neither',
    )
    kwargs.update(vkwargs)
    da.plot(**kwargs)

    if a is not None:
        #ds['vel'] = np.sqrt(ds.ugos**2 + ds.vgos**2)
        #_ds = ds.where(ds.vel < 1)
        assert isinstance(a, tuple), 'a must be a tuple'
        dsa = ds.isel(num_lines=slice(0, None, a_di), num_pixels=slice(0, None, a_di))
        u, v = dsa[a[0]], dsa[a[1]]
        if len(a)>2 and a[2] is not None:
            vel = np.sqrt(u**2 + v**2)
            ds = ds.where(vel < a[2])
        kwargs=dict(transform=crs, zorder=10)
        kwargs.update(akwargs)
        q = ax.quiver(dsa.longitude, dsa.latitude, u, v, **kwargs)
        qk = plt.quiverkey(
            q, 0.8, 0.92, a_ref,
            f'{a_ref:.2f} m/s',
            labelpos='N',
            transform=crs,
            color='k',
        )

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, zorder=5)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(ds.time.dt.strftime('%Y/%d/%m %H:%M').values[0])

    fig.savefig(os.path.join(fig_dir, f'{cycle}_{pass_number}.png'))
    if close:
        plt.close(fig)
    else:
        return fig, ax

# ---------------------- joint loader --------------------------------

def load_swot_da(
    zds, v, level='L3', 
    extent=None,
    geoselection=None, # for L2 zcoll
    project=None,
    mask=None,
    s0_rescale=True,
    dev=False,
):
    '''  joint L3, L2 loader for single variable '''

    # extract cycle and pass numbers from zds
    from scipy.stats import mode
    cycle_number = mode(zds['cycle_number'].compute())[0]
    pass_number = mode(zds['pass_number'].compute())[0]  
    
    ## load datasets
    
    ## L3 is allways loaded
    ds = zds.to_xarray()
    variables = l3['core']
    if v=='ssh':
        variables += l3['ssh']
    elif v=='sig0':
        variables += l3['sig0']
    else:
        if v in ds:
            variables += [v]
    if not dev:
        ds = ds[variables]
    if extent is not None:
        ds = zoom(ds, extent)
    if ds is None:
        return
    if project is not None:
        assert extent, 'project and extent should be specified together (wiser)'
        # add local projection
        if isinstance(project, bool):
            ds = add_geo_projections(ds)
            gproj = {
                x: ds.attrs['proj_'+x] for x in ['lonc', 'latc', 'phi']
            }
        else:
            ds = add_geo_projections(ds, **project)
            gproj = project
    ds = ds.set_coords(['time'])
    ds_l3 = ds
        
    ## L2
    if level=='L2_unsmoothed_std':
        ds = load_l2_std(
            'Unsmoothed', 
            pass_number, cycle_number, 
            extent=extent, 
            s0_rescale=s0_rescale,
        )
        if ds is None:
            return
        if project is not None:
            ds = add_geo_projections(ds, **gproj)
        ds_l2 = ds
    elif level=='L2_unsmoothed_cvl':
        if not dev:
            if v in ['ssh', 'sig0']:
                variables=l2[v]
            else:
                variables=[v]
        else:
            variables=None
        zds = load_l2(
            variables=variables,
            half_orbits=False,
            pass_numbers=pass_number,
            cycle_numbers=cycle_number,
            geoselection=geoselection,
        )
        if zds is None:
            return
        ds = zds.to_xarray()
        ds0 = ds
        if extent is not None:
            ds = zoom(ds, extent)
        if ds is None:
            return
        if project is not None:
            ds = add_geo_projections(ds, **gproj)
        ds = ds.set_coords(['time', 'longitude', 'latitude'])
        ds_l2 = ds

    ## reduce to data arrays
    da_mask = None
    if level=='L3':
        if v=="sig0" and s0_rescale:
            rescale_sigma0(ds_l3)
        if v=='ssh':
            da = (
                ds_l3['ssha_karin_2_filtered'] 
                + ds_l3['cvl_mean_dynamic_topography_cnes_cls_22']
            )
            da_mask = ds_l3['ssha_karin_2_qual']
        elif v=='sig0':
            da = ds_l3['sig0_karin_2']
            #da_mask = ds_l3['sig0_karin_2_qual'] # not available
            # no qual flag for sig0
            #if mask is not None:
            #    da = da.where(ds_l3[v+'_qual']<mask)
        else:
            da = ds_l3[v]
            if mask is not None and v+'_qual' in ds_l3:
                da_mask = ds_l3[v+'_qual']
        if da_mask is not None:
            da = da.assign_coords(mask=da_mask)
            if mask is not None:
                da = da.where(da_mask<mask)
        # time min/max
        time_min = pd.to_datetime(np.nanmin(da.time.values))
        time_max = pd.to_datetime(np.nanmin(da.time.values))
    elif level=='L2_unsmoothed_std':
        if v=='ssh':
            mdt = ds_l3['cvl_mean_dynamic_topography_cnes_cls_22']
            xcal = ds_l3['xcal']
            def _extract(ds):
                ds = ds.set_coords(['time'])
                interp_da(mdt, ds) # inplace
                interp_da(xcal, ds) # inplace
                da = (
                    ds['ssh_karin_2']
                    - ds['mean_sea_surface_cnescls']
                    + ds['xcal']
                    + ds['cvl_mean_dynamic_topography_cnes_cls_22']
                )
                # could also try to align with L3 median ...
                #da = da - da.median('num_lines') # manual cross-over correction
                da_mask = ds['ssh_karin_2_qual']
                da = da.assign_coords(mask=da_mask)
                if mask is not None:
                    da = da.where(da_mask<mask)
                return da
        elif v=='sig0':
            def _extract(ds):
                ds = ds.set_coords(['time'])
                da = ds['sig0_karin_2']
                # could also try to align with L3 median ...
                #da = da - da.median('num_lines') # manual cross-over correction
                da_mask = ds['sig0_karin_2_qual']
                da = da.assign_coords(mask=da_mask)
                if mask is not None:
                    da = da.where(da_mask<mask)
                return da            
        else:
            def _extract(ds): 
                da = ds.set_coords(['time'])[v]
                if v+'_qual' in ds:
                    da_mask = ds[v+'_qual']
                    da = da.assign_coords(mask=da_mask)
                    if mask is not None:
                        da = da.where(da_mask<=mask)
                return da
        da = {side: _extract(ds) for side, ds in ds_l2.items()}
        # time min/max
        time_min = pd.to_datetime(np.nanmin(da['left'].time.values))
        time_max = pd.to_datetime(np.nanmin(da['left'].time.values))
    elif level=='L2_unsmoothed_cvl':
        if v=="sig0" and s0_rescale:
            rescale_sigma0(ds_l2)
        if v=='ssh':
            da = (
                ds_l2['ssh_karin_2']
                - ds_l2['cvl_mean_sea_surface_cnes_22_hybrid']   #  'mean_sea_surface_cnescls'
                + ds_l2['cvl_mean_dynamic_topography_cnes_cls_22']
                - ds_l2['cvl_dac']
                - ds_l2['cvl_solid_earth_tide'] - ds_l2['cvl_ocean_tide_fes_2022'] 
                - ds_l2['cvl_internal_tide_hret'] - ds_l2['cvl_pole_tide']
                + ds_l2['cvl_height_cor_l3'] + ds_l2['cvl_phase_screen_l3'] # corrections from L3
                #+ ds_l2['cvl_height_cor_xover'] 
                # 'cvl_geoid', 
                # 'cvl_height_cor_l3', 'cvl_phase_screen_l3', # need to ask
            )
            da_mask = ds_l2["ssh_karin_2_qual"]
        elif v=='sig0':
            da = ds_l2['sig0_karin_2']
            da_mask = ds_l2["sig0_karin_2_qual"]
        else:
            da = ds_l2[v]
            if mask is not None and v+'_qual' in ds:
                da_mask = ds[v+'_qual']
        if da_mask is not None:
            da = da.assign_coords(mask=da_mask)
            if mask is not None:
                da = da.where(da_mask<mask)
        # time min/max
        time_min = pd.to_datetime(np.nanmin(da.time.values))
        time_max = pd.to_datetime(np.nanmin(da.time.values))

    if project is not None:
        if isinstance(da, dict):
            for s, ds in da.items():
                ds.attrs.update(**gproj)
        else:
           da.attrs.update(**gproj)
    if dev:
        if level=='L3':
            return da, time_min, time_max, ds_l3
        else:
            return da, time_min, time_max, ds_l2, ds_l3
    return da, time_min, time_max
    

# ---------------------- ERA5 --------------------------------

def _zoom_era5_extent(ds, extent):
    lon0, lon1, lat0, lat1 = extent
    lon0 = lon0%360
    lon1 = lon1%360
    #assert extent[0]>=0 or extent[1]>=0, 'longitudes should be within [0,360]'
    return ds.sel(
        longitude=slice(lon0, lon1), 
        latitude=slice(lat1, lat0), # lat are from 90 to -90 in era5
    )
    
def _zoom_era5_location(ds, location):
    lon, lat = location
    lon = lon%360
    #assert location[0]>=0, 'longitude should be within [0,360]'
    return ds.sel(
        longitude=lon, 
        latitude=lat,
        method='nearest',
    )
    
    
def load_era5(start, end=None, extent=None, location=None):
    ''' load era5 data
    '''
    data_dir = '/work/HELPDESK_SWOTLR/swot_diego/era5'

    if end is None:
        end = start
    if isinstance(start, str):
        start = pd.Timestamp(start)
    if isinstance(end, str):
        end = pd.Timestamp(end)
        
    files = [
        glob(os.path.join(data_dir, f'*{t.strftime("%Y%m%d")}.nc'))[0] for t in pd.date_range(start, end, freq='1d')
    ]
    
    preprocess=None
    assert extent is None or location is None, 'extent and location cannot be true simultaneously'
    if extent is not None:
        preprocess = lambda ds: _zoom_era5_extent(ds, extent)
    if location is not None:
        preprocess = lambda ds: _zoom_era5_location(ds, location)
    
    ds = xr.open_mfdataset(files, preprocess=preprocess)
    ds['wind_speed'] = np.sqrt(ds['u10']**2+ds['v10']**2)
    ds['wind_stress'] = np.sqrt(ds['iews']**2+ds['inss']**2)
    
    return ds

# wind

def show_wind_teries(era5, label='', ax=None):

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(10,4))
        
    era5['wind_speed'].plot(ax=ax, color='k', lw=3, label='wind speed')
    era5['u10'].plot(ax=ax, label='eastward')
    era5['v10'].plot(ax=ax, label='northward')
    #ax.axvline(t, color='k', lw=3)
    ax.grid()
    ax.legend()
    ax.set_ylabel('[m/s]')
    label = label+f' lon={float(era5.longitude):.2f}, lat={float(era5.latitude):.2f}'
    ax.set_title(label)
    

def get_wind_hist(era5, bins):
    h, xb, yb = np.histogram2d(era5['u10'], era5['v10'], bins=bins, density=True)
    xc, yc = (xb[:-1]+xb[1:])*.5, (yb[:-1]+yb[1:])*.5

    ds = xr.Dataset(dict(h=(('y','x'), h.T)), coords=dict(x=('x', xc), y=('y', yc)))
    ds = ds.rename(x='u10', y='v10')
    return ds['h']


def show_wind_hist(era5, bins_speed, bins, label=''):
            
    # wind speed
    fig, ax = plt.subplots(1,1, figsize=(5,3))
    era5['wind_speed'].plot.hist(ax=ax, bins=bins_speed, color='k', lw=3, label='wind speed')
    ax.grid()
    ax.legend()
    ax.set_ylabel('')
    ax.set_xlabel('[m/s]')
    label = label+f' lon={float(era5.longitude):.2f}, lat={float(era5.latitude):.2f}'
    ax.set_title(label)

    
    # polar plot histogram
    h = get_wind_hist(era5, bins)

    fig = plt.figure(figsize=(5,5))

    ax = fig.add_subplot()
    axp = fig.add_axes(ax.get_position().bounds, polar=True, frameon=False)

    h.plot(ax=ax, cmap='Greys', add_colorbar=False)

    rmax = bins[-1]
    ax.set_aspect('equal')
    ax.set_xlim(-rmax,rmax)
    ax.set_ylim(-rmax,rmax)
    axp.set_rlim(0,rmax)
    ax.axis('off')
    
def plot_wind_tseries_small(era5):
    fig, ax = plt.subplots(1,1)
    era5['wind_speed'].plot(ax=ax, color='k', lw=3, label='wind speed')
    era5['u10'].plot(ax=ax, label='u10')
    era5['v10'].plot(ax=ax, label='v10')
    ax.axvline(t, color='k', lw=3)
    ax.grid()
    ax.legend()
    ax.set_ylabel('[m/s]')
    ax.set_title('')    

# ---------------------- areas of interest --------------------------------


def get_extent(coordinates):
    c=np.array(coordinates).squeeze()
    extent = [c[:,0].min(), c[:,0].max(), c[:,1].min(), c[:,1].max()]
    return extent

regions = {}

### North-West Med

# https://www.keene.edu/campus/maps/tool/?coordinates=4.9987410%2C%2043.5923275%0A3.3078806%2C%2040.1452893%0A5.1870042%2C%2039.5802903%0A6.9452485%2C%2043.2612061%0A4.9987410%2C%2043.5923275
regions['med_NW'] = {
  'coordinates': [
    [
      [
        5.0441468,
        43.5963059
      ],
      [
        3.3078806,
        40.1452893
      ],
      [
        5.1870042,
        39.5802903
      ],
      [
        6.9452485,
        43.2612061
      ],
      [
        5.0441468,
        43.5963059
      ]
    ]
  ],
  'type': 'Polygon'
}
regions['med_NW']['passes'] = [3]
regions['med_NW']['xlims'] = [-70, 70] #km
regions['med_NW']['ylims'] = [-220, 200] #km

### North-West Med - North Current

# https://www.keene.edu/campus/maps/tool/?coordinates=4.6416377%2C%2043.6241471%0A4.1362878%2C%2042.3057530%0A6.1413098%2C%2041.9022770%0A6.7400634%2C%2043.2131833%0A4.6416377%2C%2043.6241471
regions['med_NW_CN'] = {
  'coordinates': [
    [
      [
        4.6416377,
        43.6241471
      ],
      [
        4.1362878,
        42.305753
      ],
      [
        6.1413098,
        41.902277
      ],
      [
        6.7400634,
        43.2131833
      ],
      [
        4.6416377,
        43.6241471
      ]
    ]
  ],
  'type': 'Polygon'
}
regions['med_NW_CN']['passes'] = [3]
regions['med_NW_CN']['xlims'] = [-65, 65] #km
regions['med_NW_CN']['ylims'] = [-80, 80] #km


### Mascarene East

# https://www.keene.edu/campus/maps/tool/?coordinates=61.3696289%2C%20-10.6390138%0A61.3696289%2C%20-10.6390138%0A62.1386719%2C%20-19.3733407%0A70.0048828%2C%20-19.3422450%0A69.7521973%2C%20-10.6822006%0A61.3696289%2C%20-10.6390138
regions['mascarene'] = {
  'coordinates': [
    [
      [
        61.3696289,
        -10.6390138
      ],
      [
        61.3696289,
        -10.6390138
      ],
      [
        62.1386719,
        -19.3733407
      ],
      [
        70.0048828,
        -19.342245
      ],
      [
        69.7521973,
        -10.6822006
      ],
      [
        61.3696289,
        -10.6390138
      ]
    ]
  ],
  'type': 'Polygon'
}
regions['mascarene']['passes'] = [12]

### Amazone area

regions['amazone'] = {
  'coordinates': [
    [
      [
        -52.0532227,
        15.8345357
      ],
      [
        -51.2841797,
        -1.6257584
      ],
      [
        -36.7382813,
        -5.7690359
      ],
      [
        -30.7617188,
        14.7961276
      ],
      [
        -52.0532227,
        15.8345357
      ]
    ]
  ],
  'type': 'Polygon'
}
regions['amazone']['passes'] = [7, 20]

### Maluku
# https://www.keene.edu/campus/maps/tool/?coordinates=124.9914551%2C%202.0375343%0A124.1070557%2C%20-1.8453840%0A126.3922119%2C%20-1.9826364%0A127.1832275%2C%201.9112665%0A124.9914551%2C%202.0375343
regions['maluku'] = {
  'coordinates': [
    [
      [
        124.9914551,
        2.0375343
      ],
      [
        124.1070557,
        -1.845384
      ],
      [
        126.3922119,
        -1.9826364
      ],
      [
        127.1832275,
        1.9112665
      ],
      [
        124.9914551,
        2.0375343
      ]
    ]
  ],
  'type': 'Polygon'
}
regions['maluku']['passes'] = [21]
regions['maluku']['xlims'] = [-65, 65] #km
regions['maluku']['ylims'] = [-200, 200] #km
regions['maluku']['bathy'] = "/work/HELPDESK_SWOTLR/swot_diego/maluku/bathy/gebco_2023_n2.4594_s-2.1999_w123.7815_e128.2801.nc"
regions['maluku']['tides'] = "/work/HELPDESK_SWOTLR/swot_diego/maluku/tides/fes2014_mooring_time_series.nc"
regions['maluku']['tides_harmonics'] = "/work/HELPDESK_SWOTLR/swot_diego/maluku/tides/fes2014_harmonics.nc"


## California - Nico
# https://www.keene.edu/campus/maps/tool/?coordinates=4.6416377%2C%2043.6241471%0A4.1362878%2C%2042.3057530%0A6.1413098%2C%2041.9022770%0A6.7400634%2C%2043.2131833%0A4.6416377%2C%2043.6241471
regions['cali'] = {
  "coordinates": [
    [
      [
        -126.4086415,
        35.012002
      ],
      [
        -122.3875218,
        35.0030034
      ],
      [
        -122.3648577,
        38.8995834
      ],
      [
        -126.5296725,
        38.8995834
      ],
      [
        -126.4197829,
        35.0299964
      ]
    ]
  ],
  "type": "LineString"
}
regions['cali']['passes'] = [13, 26]
#regions['cali']['passes'] = [26,]

# collect coordinates and central location
for r, info in regions.items():
    e =  get_extent(info['coordinates'])
    regions[r]['location'] =  [(e[0]+e[1])*.5, (e[2]+e[3])*.5]
    regions[r]['extent'] = e
