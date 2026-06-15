import h5py
import numpy as np
import astropy.units as u
import astropy.constants as const
import pylab

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.colors as mpl_c
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


GAMMA      = 5./3
GAMMA_CR   = 4./3
k_B        = 1.381e-16
m_p        = 1.66e-24
HYDROGENMASS_FRAC = 0.76

import sys, os
sys.path.insert(0, '/cosma8/data/dp317/dc-naza3/arepo-snap-util')
import arepo_run as arun
import gadget

def _count_snaps(path):
    n = 0
    while os.path.exists(os.path.join(path, f'snap_{n:03d}.hdf5')):
        n += 1
    return n

def number_string(snap):
    snap_str = str(snap)
    if len(snap_str) < 3:
        snap_str = (3-len(snap_str))*str(0)+snap_str
    return snap_str

def calc_snap_time(s):
    Length_unit = s.header['UnitLength_in_cm']
    Velocity_unit = s.header['UnitVelocity_in_cm_per_s'] 
    Time_unit = Length_unit / Velocity_unit
    Time = s.header['Time'] * Time_unit
    Time_Myr = (Time * u.s).to(u.Myr).value
    return Time_Myr


def get_vranges(snap_nums,snappath=None,quant='temp',flag_excl=True):
    minval = np.empty(0)
    maxval = np.empty(0)

    for i_s, num in enumerate(snap_nums):
        s = load_snap_data(num,snappath=output_path,snapbase=SNAPBASE)
        quant_data = s.data[quant]
        if flag_excl: #Exclude the BOLA cells
            quant_data = quant_data[s.data['flag']==0]
        minval = np.append(minval, quant_data.min())
        maxval = np.append(maxval, quant_data.max())
        #print('\n',s.data[quant].min(),s.data[quant].max())

    ranges = [minval.min(),maxval.max()]
    print('Min/max values at snaps:',minval.argmin(),minval.argmax())
    return ranges


def calc_mu(data,ionisation='ionised'):
    X = HYDROGENMASS_FRAC

    if 'metals' in data.keys():
        Z = data['metals'] ## metal mass fraction
    else:
        Z = 0

    if 'ne' in data.keys():
        ne = data['ne']
        print('\t>>Electrion density found, calculating mu...')
    elif ionisation == 'ionised':
        ne = np.ones_like(data['u']) * 2 / (1+X)
        print('\t>>Electron density not found, asssuming default ionisation: {}'.format(ionisation))
    elif ionisation =='neutral':
        ne = np.zeros_like(data['u'])
        print('\t>>Electron density not found, asssuming default ionisation: {}'.format(ionisation))

    mu = 4 / (X * (3 + 4*ne) + 1 - Z)
    return mu

def calc_T(data, unit_v):
    temp = data['u']*(unit_v)**2 * (GAMMA - 1) * data['mu']  * m_p  / k_B
    print(temp)
    return temp

def calc_P(data):
    pressure = (data['rho']) * (data['u']) * (GAMMA - 1)
    print('pressure:', pressure)
    return pressure

def calc_CRP(data, gamma_cr=4/3):
    pressure = (data['rho']) * (data['cren']) * (gamma_cr - 1)
    print('CR pressure:', pressure)
    return pressure

def rho_to_n_cm(data, unit_rho):
    n_cms = data['rho'] * unit_rho/ (m_p )
    return n_cms

def rho_to_nH_cm(data, unit_rho):
    nH_cm = HYDROGENMASS_FRAC * data['rho'] * unit_rho/ (m_p )
    return nH_cm


def calc_cooling(data, header, verbose=False):
    if 'coor' in data:
        ## Setup unit system
        length_unit = header['UnitLength_in_cm'] * u.cm
        velocity_unit = header['UnitVelocity_in_cm_per_s'] * u.cm / u.s
        mass_unit = header['UnitMass_in_g'] * u.g
        time_unit = length_unit / velocity_unit

        ## s.data['coor'] is the cooling rate per unit mass, in simulations units

        cool_time =  (time_unit * data['u'] / data['coor']).to(u.Myr)  ## Comes from io_fields.c
        cool_rate_mass = (data['coor'] * velocity_unit**3 / length_unit).to(u.erg / u.s / u.g)

        ratefact = (HYDROGENMASS_FRAC / (m_p * mass_unit))**2 * data['rho'] * mass_unit / length_unit**3
        cool_rate_volume = (cool_rate_mass/ratefact).to(u.erg/u.s * u.cm**3)

        return cool_time.value, cool_rate_mass.value, cool_rate_volume.value
    
    else:
        if verbose:
            print('\t>>Field COOR not found, skipping cooling calculation...')
        return None, None, None
    

def calc_bremsstrahlung(data,T_cut=None,gaunt_factor=1, Z_factor = 1):
    # See Sijacki & Springel 06, also Bennett
    # Bolometric X-ray (all bands)

    xray = np.zeros_like(data['temp'])

    vols = (data['vol']*u.kpc**3).to(u.cm**3).value
    xray =  1.4e-27 * gaunt_factor * Z_factor**2 * data['temp']**(0.5) * data['ne'] * data['nH_cm']**2 * (1-data['nh']) * vols

    if T_cut is not None:
        xray[data['temp']<T_cut] = 0

    return xray


def load_snap_data(num,snappath=None,snapbase='snap_',advanced_xrays=False,verbose=False,default_ionisation='ionised',norm=False, ic=False):
    
    snapname = snappath + snapbase + number_string(num)
    print('\n>Loading snap',snapname)

    if not ic:
        o = arun.Run(snappath=snappath,snapbase=snapbase)
        s = o.loadSnap(snapnum=num)
        unit_v = s.header['UnitVelocity_in_cm_per_s']
        unit_l = s.header['UnitLength_in_cm'] 
        unit_m = s.header['UnitMass_in_g']
    else:
        import gadget_snap as gsnap
        s = gsnap.gadget_snapshot(filename=snappath, hdf5=True)
        unit_v =1.651077e6
        unit_l = 3.0857e21
        unit_m = 1.9885e43


    unit_t = unit_l / unit_v
    unit_rho = unit_m / unit_l**3

    s.data['mu'] = calc_mu(s.data,ionisation=default_ionisation)
    s.data['temp'] = calc_T(s.data, unit_v=unit_v)

    if 'pres' not in s.data:
        s.data['pres'] = calc_P(s.data)

    if 'wind' not in s.data and 'pass' in s.data:
        s.data['wind'] = s.data['pass']

    s.data['n_dens_cm'] = rho_to_n_cm(s.data, unit_rho=unit_rho)

    s.data['nH_cm'] = rho_to_nH_cm(s.data, unit_rho=unit_rho)
    xx = s.data['pos'][:,0] - s.boxsize/2.
    yy = s.data['pos'][:,1] - s.boxsize/2.
    zz = s.data['pos'][:,2] - s.boxsize/2.
    rr = np.sqrt(xx**2 + yy**2 + zz**2)
    s.data['r'] = rr

    vx = s.data['vel'][:,0]
    vy = s.data['vel'][:,1]
    vz = s.data['vel'][:,2]
    s.data['vrad'] = (vx * xx/rr + vy * yy/rr + vz * zz/rr)

    s.data['speed']     = np.linalg.norm(s.data['vel'], axis=1)

    s.data['energdens'] = s.data['u']*s.data['rho']

    # Unit conversion factors from code units to cgs
    # B-field units: Gauss (assuming Gadget/AREPO default cosmological/comoving units check, though here we apply cgs factors for energy density)
    unit_b = np.sqrt(unit_rho * unit_v**2)

    if 'bfld' in s.data:
        # B-field specific energy (energy per unit mass in code units -> cgs)
        b_cgs = s.data['bfld'] * unit_b
        rho_cgs = s.data['rho'] * unit_rho
        s.data['bflden'] = np.sum(b_cgs**2, axis=1) / (8 * np.pi * rho_cgs)/(unit_v)**2  # Convert to energy per unit mass in code units    
        s.data['bflds'] = np.linalg.norm(s.data['bfld'], axis=1)
        s.data['bfldpres'] = s.data['rho'] * s.data['bflden']
        s.data['xb'] = (s.data['bfldpres'])/ (s.data['pres'])
        s.data['beta'] = 1/s.data['xb']
    if 'cren' in s.data:
        s.data['crpres'] = calc_CRP(s.data)
        s.data['xcr'] = s.data['crpres'] / s.data['pres']
        s.data['crendens'] = s.data['cren'] * s.data['rho']
        s.data['ecth'] = s.data['cren'] / s.data['u']
        s.data['crener'] = s.data['cren'] * s.data['mass']
    
    if 'xb' in s.data and 'xcr' in s.data:
        s.data['xbcr'] = s.data['xb'] / s.data['xcr']

    if 'coor' in s.data:
        s.data['cool_time'], s.data['cool_rate_mass'], s.data['cool_rate_volume'] = calc_cooling(s.data, s.header)
    else:
        print('\t>>Field COOR not found, skipping cooling calculation...')
 
    if 'ne' in s.data:
        s.data['ne_cm'] = s.data['ne'] * s.data['nH_cm']
  
        s.data['xray'] = calc_bremsstrahlung(s.data)
    
    if norm:
        s0 = o.loadSnap(snapnum=0)
        for data_key in [k for k in s0.data.keys() if k != 'pos']:  # Don't normalize positions
            # Avoid division by zero and cast to float to avoid UFuncTypeError
            div = np.median(s0.data[data_key]) if np.median(s0.data[data_key]) != 0 else 1e-10
            if data_key == 'cren': div = np.median(s0.data['u'])  # Normalize CR energy by initial internal energy, not CR energy:
            s.data[data_key] = s.data[data_key] / div
            print(f'Normalized {data_key} by dividing by median value: {div}')

    return s

def radial_profile_lin(s, field, r_range=(2.5, 200), nbins=200, post_shock=False, shock_path=None, unit_rho=1):
    """Linearly binned radial profile.  Returns (r_centres, profile) or (None, None)."""
    if field not in s.data:
        return None, None
    if post_shock:
        with h5py.File(shock_path, "r") as shocks_file:
            s.data['shocks_coords']     = shocks_file["Coordinates"][:]
            s.data['temperature']       = shocks_file["Temperature"][:]
            s.data['preshock_temp']     = shocks_file["PreShockTemperature"][:]
            s.data['mach']              = shocks_file["Machnumber"][:]
            s.data['shock_direction']   = shocks_file["ShockDirection"][:]
            s.data['preshock_rho']      = shocks_file["PreShockDensity"][:] * unit_rho
            s.data['postshock_rho']     = shocks_file["PostShockDensity"][:] * unit_rho
            s.data['preshock_p']        = shocks_file["PreShockPressure"][:]
            s.data['postshock_p']       = shocks_file["PostShockPressure"][:]
            s.data['preshock_v']        = shocks_file["PreShockVelocity"][:]
            s.data['postshock_v']       = shocks_file["PostShockVelocity"][:]
            s.data['surf']              = shocks_file["Surface"][:]    
            s.data['uflux']         = shocks_file["GeneratedInternalEnergyFlux"][:]
            s.data['edis']          = s.data['uflux']*s.data['surf'] 
        
        # Find the mapping from shocks_coords to pos
        from scipy.spatial import cKDTree

        # Build a KDTree for fast nearest neighbor lookup
        tree = cKDTree(s.data['shocks_coords'])
        _, indices = tree.query(s.data['pos'])

        # Reorder all shock parameters to match pos ordering
        shock_params = ['temperature', 'preshock_temp', 'mach', 'shock_direction', 
                        'preshock_rho', 'postshock_rho', 'preshock_p', 'postshock_p',
                        'preshock_v', 'postshock_v', 'surf', 'uflux', 'edis']

        for param in shock_params:
            if param in s.data:
                s.data[param] = s.data[param][indices]

        # Now shocks_coords should match pos
        s.data['shocks_coords'] = s.data['shocks_coords'][indices]
    pos  = s.data['pos']
    ctr  = np.array([s.boxsize / 2] * 3)
    r    = np.linalg.norm(pos - ctr, axis=1)
    vals = s.data[field]
    rlo, rhi = r_range
    mask = (r >= rlo) & (r <= rhi) & np.isfinite(vals)
    r_bins  = np.linspace(rlo, rhi, nbins + 1)
    r_ctrs  = 0.5 * (r_bins[:-1] + r_bins[1:])
    sums, _ = np.histogram(r[mask], bins=r_bins, weights=vals[mask])
    cnts, _ = np.histogram(r[mask], bins=r_bins)
    with np.errstate(invalid='ignore'):
        profile = np.where(cnts > 0, sums / cnts, np.nan)
    return r_ctrs, profile

def radial_profile_log(s, field, r_range=(2.5, 200), nbins=200):
    """
    Logarithmically binned radial profile.  Returns (r_centres, values).
    Mirrors rvsval() from shocks1d.ipynb.
    """
    pos = s.data['pos']
    ctr = np.array([s.boxsize / 2] * 3)
    r   = np.linalg.norm(pos - ctr, axis=1)

    # radial velocity (attach once)
    if 'vrad' not in s.data:
        diff = pos - ctr
        vdot = np.sum(diff * s.data['vel'], axis=1)
        s.data['vrad'] = vdot / (r + 1e-30)

    if field not in s.data:
        return None, None

    vals     = s.data[field]
    rlo, rhi = r_range
    mask     = (r >= rlo) & (r <= rhi) & np.isfinite(vals)

    r_bins   = np.logspace(np.log10(rlo), np.log10(rhi), nbins + 1)
    r_ctrs   = 0.5 * (r_bins[:-1] + r_bins[1:])
    idx      = np.digitize(r[mask], r_bins)
    profile  = np.array([
        vals[mask][idx == i].mean() if np.any(idx == i) else np.nan
        for i in range(1, nbins + 1)
    ])
    return r_ctrs, profile


def plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [0,0],
        var = 'temp',
        weighted = 'rho', # or None
        ranges = None,
        cmap = 'viridis',
        logplot = True,
        divzero = False,
        divzero_centre = None,

        image_proj = 'side',
        proj_on = True,
        proj_fact = 0.1,
        res = 258,
        plotsize = 2,
        colorbar = True,
        add_vec = False,
        vec_val = 'bfld',
        numthreads = 1
        ):


    ## Set up subplot
    quad_ax = plt.Subplot(fig, quad_subs[quad_ax_loc[0],quad_ax_loc[1]])
    fig.add_subplot(quad_ax)

    ## Calculate centre and set up viewing angle
    x_pol = -1 + 2*quad_ax_loc[1]
    y_pol = 1 - 2*quad_ax_loc[0]

    x_centre = s.boxsize/2 + x_pol*plotsize/2
    y_centre = s.boxsize/2 + y_pol*plotsize/2
    z_centre = s.boxsize/2

    if image_proj == 'side':
        plot_centre = [x_centre, z_centre, y_centre]
        axes_sum = [0,2]
    elif image_proj == 'top':
        plot_centre = [x_centre, y_centre, z_centre]
        axes_sum = [0,1]

    print(plot_centre)


    ## Call gadget_snap from arepo-snap-utils to plot in given axis

    if weighted is None:
        s.axplot_Aslice(quad_ax,value=var,cmap=cmap,colorbar=colorbar,divzero=divzero,divzero_centre=divzero_centre,vrange=ranges,axes=axes_sum,logplot=logplot,box=[plotsize,plotsize],center=plot_centre,proj=proj_on,proj_fact=proj_fact,res=res, numthreads=numthreads)
    else:
        s.axplot_Aweightedslice(quad_ax,value=var,weights=weighted,cmap=cmap,colorbar=colorbar,divzero=divzero,divzero_centre=divzero_centre,vrange=ranges,axes=axes_sum,logplot=logplot,box=[plotsize,plotsize],center=plot_centre,proj=proj_on,proj_fact=proj_fact,res=res, numthreads=numthreads)

    ## Set axis labels relative to centre of box

    quad_ax.set_xticklabels(np.round(quad_ax.get_xticks()-s.boxsize/2, decimals=2))
    quad_ax.set_yticklabels(np.round(quad_ax.get_yticks()-s.boxsize/2, decimals=2))

    ## Tidy up axis edges and labels

    if quad_ax_loc[0] == 0:
        quad_ax.spines['bottom'].set_visible(False)
        quad_ax.set_xticks([])
    if quad_ax_loc[0] == 1:
        quad_ax.spines['top'].set_visible(False)
    if quad_ax_loc[1] == 1:
        quad_ax.spines['left'].set_visible(False)
        quad_ax.set_yticks([])

    # Get the last QuadMesh (pcolormesh) from the axis - this is the mappable
    mappable = quad_ax.collections[-1] if quad_ax.collections else None
    
    if add_vec:
        # Add magnetic field streamlines from a simple gridded in-slice field.
        # half = plotsize / 2
        # x_min = plot_centre[axes_sum[0]] - half
        # x_max = plot_centre[axes_sum[0]] + half
        # y_min = plot_centre[axes_sum[1]] - half
        # y_max = plot_centre[axes_sum[1]] + half
        # # x_min, x_max = quad_ax.get_xlim()
        # # y_min, y_max = quad_ax.get_ylim()

        # # Use the full LOS within this panel window to avoid sparse streamline coverage.
        # mask_x = (s.pos[:, axes_sum[0]] >= x_min) & (s.pos[:, axes_sum[0]] <= x_max)
        # mask_y = (s.pos[:, axes_sum[1]] >= y_min) & (s.pos[:, axes_sum[1]] <= y_max)
        # mask = mask_x & mask_y

        # if np.sum(mask) > 10:
        #     pos = s.pos[mask]
        #     vec = s.data[vec_val][mask]

        #     x = pos[:, axes_sum[0]]
        #     y = pos[:, axes_sum[1]]
        #     u = vec[:, axes_sum[0]]
        #     v = vec[:, axes_sum[1]]

        #     x_grid = np.linspace(x_min, x_max, 32)
        #     y_grid = np.linspace(y_min, y_max, 32)
        #     X, Y = np.meshgrid(x_grid, y_grid)

        #     from scipy.interpolate import griddata

        #     # Interpolate scattered particle data directly onto a regular grid
        #     U = griddata((x, y), u, (X, Y), method='nearest')
        #     V = griddata((x, y), v, (X, Y), method='nearest')
        # Add magnetic field streamlines from a gridded in-slice field.
        out_axis = [i for i in [0,1,2] if i not in axes_sum][0]
        slice_val = plot_centre[out_axis]
        dz = plotsize * 0.05  # 5% thickness for selecting particles near the slice

        # Select particles in the slice and within the quadrant bounding box
        mask_out = np.abs(s.pos[:, out_axis] - slice_val) < dz
        mask_x = np.abs(s.pos[:, axes_sum[0]] - plot_centre[axes_sum[0]]) < (plotsize/2)
        mask_y = np.abs(s.pos[:, axes_sum[1]] - plot_centre[axes_sum[1]]) < (plotsize/2)
        mask = mask_out & mask_x & mask_y


        # Subsample particles further to avoid cluttered quiver plot
        if np.sum(mask) > 0:
            sample_step = max(1, int(np.sum(mask) / 500))  # Target ~500 vectors
            sampled_pos = s.pos[mask][::sample_step]
            sampled_B = s.data[vec_val][mask][::sample_step]

            x_pos = sampled_pos[:, axes_sum[0]]
            y_pos = sampled_pos[:, axes_sum[1]]
            u_v = sampled_B[:, axes_sum[0]]
            v_v = sampled_B[:, axes_sum[1]]

            # quad_ax.quiver(x_pos, y_pos, u_v, v_v, color='gray',
            #             alpha=0.8, scale=None, pivot='mid', headwidth=3, headlength=4)

            
            
            pos = s.pos#[mask]
            vec = s.data[vec_val]#[mask]

            x = pos[:, axes_sum[0]]
            y = pos[:, axes_sum[1]] 
            u = vec[:, axes_sum[0]]
            v = vec[:, axes_sum[1]]

            half = plotsize / 2.0
            x_min = plot_centre[axes_sum[0]] - half
            x_max = plot_centre[axes_sum[0]] + half
            y_min = plot_centre[axes_sum[1]] - half
            y_max = plot_centre[axes_sum[1]] + half
            print(f"Streamline grid bounds: x=({x_min:.2f}, {x_max:.2f}), y=({y_min:.2f}, {y_max:.2f})")
            x_grid = np.linspace(x_min, x_max, 256)
            y_grid = np.linspace(y_min, y_max, 256)
            X, Y = np.meshgrid(x_grid, y_grid)

            from scipy.interpolate import griddata
            import lic

            U = griddata((x, y), u, (X, Y), method='nearest')
            V = griddata((x, y), v, (X, Y), method='nearest')

            # Don't zero out NaNs - use nearest neighbour so there shouldn't be any
            # but just in case:
            # U = np.where(np.isnan(U), np.nanmean(u), U)
            # V = np.where(np.isnan(V), np.nanmean(v), V)

            xlim = quad_ax.get_xlim()
            ylim = quad_ax.get_ylim()

            # quad_ax.streamplot(X, Y, U, V, 
            #             color='white',
            #     density=0.5,
            #     linewidth=0.5,
            #     arrowsize=0.7,
            #     zorder=6)

            # Use a smoother seed texture instead of pure white noise
            from scipy.ndimage import gaussian_filter
            seed = gaussian_filter(np.random.rand(*U.shape), sigma=1.5)    
            
            print(quad_ax.get_xlim(), quad_ax.get_ylim())
            lic_img = lic.lic(U.T, V.T, length=30, contrast=True)
            
            quad_ax.imshow(
                lic_img.T,
                origin='lower',
                extent=[x_min, x_max, y_min, y_max],
                cmap='gray',
                alpha=0.3,
                zorder=6,
                aspect='auto'
            )

            quad_ax.set_xlim(xlim)
            quad_ax.set_ylim(ylim)

    return quad_ax, mappable

def plot_tri_axis(
        s,
        fig,
        tri_ax_loc,        # 0 = left, 1 = right, 2 = bottom
        plot_rect = (0.08, 0.08, 0.84, 0.84),  # (left, bottom, width, height) in figure coords
        var = 'temp',
        weighted = 'rho',
        ranges = None,
        cmap = 'viridis',
        logplot = True,
        divzero = False,
        divzero_centre = None,

        image_proj = 'side',
        proj_on = True,
        proj_fact = 0.1,
        res = 258,
        plotsize = 2,
        colorbar = True,
        add_vec = False,
        vec_val = 'bfld',
        numthreads = 1,
        ):
    """Inverted-Y (⅄) three-panel layout within a rectangular border.

    All three axes share the same plot_rect; each renders the centre of its
    snapshot.  A polygon clip path restricts visibility to the panel's region.
    The three regions tile the rectangle exactly:

        tri_ax_loc=0  left   — quadrilateral (0,0)-(0,1)-(0.5,1)-(0.5,0.5)
        tri_ax_loc=1  right  — quadrilateral (0.5,1)-(1,1)-(1,0)-(0.5,0.5)
        tri_ax_loc=2  bottom — triangle      (0,0)-(0.5,0.5)-(1,0)

    (vertices in axes coordinates, centre of square is (0.5, 0.5))

    Draw the three dividing lines in the calling script after plotting all
    panels (vertical stem + two diagonals from the centre to the corners).
    """
    from matplotlib.patches import Polygon as MplPolygon

    L, B, W, H = plot_rect

    ## All panels are placed on the same square axes
    tri_ax = fig.add_axes([L, B, W, H])

    ## All panels show the centre of their respective simulation box
    x_centre = s.boxsize / 2
    y_centre = s.boxsize / 2
    z_centre = s.boxsize / 2

    if image_proj == 'side':
        plot_centre = [x_centre, z_centre, y_centre]
        axes_sum = [0, 2]
    elif image_proj == 'top':
        plot_centre = [x_centre, y_centre, z_centre]
        axes_sum = [0, 1]

    print(plot_centre)

    ## Render slice / projection
    if weighted is None:
        s.axplot_Aslice(tri_ax, value=var, cmap=cmap, colorbar=colorbar,
                        divzero=divzero, divzero_centre=divzero_centre, vrange=ranges,
                        axes=axes_sum, logplot=logplot, box=[plotsize, plotsize],
                        center=plot_centre, proj=proj_on, proj_fact=proj_fact,
                        res=res, numthreads=numthreads)
    else:
        s.axplot_Aweightedslice(tri_ax, value=var, weights=weighted, cmap=cmap,
                                colorbar=colorbar, divzero=divzero,
                                divzero_centre=divzero_centre, vrange=ranges,
                                axes=axes_sum, logplot=logplot, box=[plotsize, plotsize],
                                center=plot_centre, proj=proj_on, proj_fact=proj_fact,
                                res=res, numthreads=numthreads)

    ## Grab the pcolormesh mappable before the clip patch is added
    mappable = tri_ax.collections[-1] if tri_ax.collections else None

    if add_vec:
        out_axis = [i for i in [0, 1, 2] if i not in axes_sum][0]
        slice_val = plot_centre[out_axis]
        dz = plotsize * 0.05

        mask_out = np.abs(s.pos[:, out_axis] - slice_val) < dz
        mask_x = np.abs(s.pos[:, axes_sum[0]] - plot_centre[axes_sum[0]]) < (plotsize/2)
        mask_y = np.abs(s.pos[:, axes_sum[1]] - plot_centre[axes_sum[1]]) < (plotsize/2)
        mask = mask_out & mask_x & mask_y

        if np.sum(mask) > 0:
            pos = s.pos
            vec = s.data[vec_val]

            x = pos[:, axes_sum[0]]
            y = pos[:, axes_sum[1]]
            u = vec[:, axes_sum[0]]
            v = vec[:, axes_sum[1]]

            half = plotsize / 2.0
            x_min = plot_centre[axes_sum[0]] - half
            x_max = plot_centre[axes_sum[0]] + half
            y_min = plot_centre[axes_sum[1]] - half
            y_max = plot_centre[axes_sum[1]] + half
            print(f"Streamline grid bounds: x=({x_min:.2f}, {x_max:.2f}), y=({y_min:.2f}, {y_max:.2f})")

            x_grid = np.linspace(x_min, x_max, 256)
            y_grid = np.linspace(y_min, y_max, 256)
            X, Y = np.meshgrid(x_grid, y_grid)

            from scipy.interpolate import griddata
            import lic

            U = griddata((x, y), u, (X, Y), method='nearest')
            V = griddata((x, y), v, (X, Y), method='nearest')

            xlim = tri_ax.get_xlim()
            ylim = tri_ax.get_ylim()

            from scipy.ndimage import gaussian_filter
            seed = gaussian_filter(np.random.rand(*U.shape), sigma=1.5)

            print(tri_ax.get_xlim(), tri_ax.get_ylim())
            lic_img = lic.lic(U.T, V.T, length=30, contrast=True)

            tri_ax.imshow(
                lic_img.T,
                origin='lower',
                extent=[x_min, x_max, y_min, y_max],
                cmap='gray',
                alpha=0.3,
                zorder=6,
                aspect='auto'
            )

            tri_ax.set_xlim(xlim)
            tri_ax.set_ylim(ylim)

    ## Clip panel to its inverted-Y region (axes coordinates)
    # Centre of the square is (0.5, 0.5); the three polygons tile it exactly.
    clip_verts = {
        0: [[0, 0], [0, 1], [0.5, 1], [0.5, 0.5]],   # left quadrilateral
        1: [[0.5, 1], [1, 1], [1, 0], [0.5, 0.5]],    # right quadrilateral
        2: [[0, 0], [0.5, 0.5], [1, 0]],               # bottom triangle (apex up)
    }[tri_ax_loc]

    clip_patch = MplPolygon(clip_verts, transform=tri_ax.transAxes, closed=True)
    tri_ax.add_patch(clip_patch)
    clip_patch.set_visible(False)
    for coll in tri_ax.collections:
        coll.set_clip_path(clip_patch)
    for img in tri_ax.images:
        img.set_clip_path(clip_patch)

    tri_ax.set_facecolor('none')
    for spine in tri_ax.spines.values():
        spine.set_visible(False)
    # tri_ax.set_xticks([])
    # tri_ax.set_yticks([])

    return tri_ax, mappable


def find_shock_radius(s, r_range=(1e-3,1), nbins=500):
    # r_shock, t_arr = [], []
    # for n, (s, t) in enumerate(zip(snaps, times)):
    r, mach = radial_profile_lin(s, 'mach', r_range=r_range, nbins=nbins)
    idx = np.nanargmax(mach)
    r_shock = r[idx]
    
    # Find second maximum (reverse shock)
    mach_copy = mach.copy()
    right, left =0, 0
    while mach_copy[idx+right] > 0 and idx+right < len(mach_copy)-1:
        right += 1  
    while mach_copy[idx-left] > 0 and idx-left > 0:
        left += 1
    mach_copy[max(0, idx-left):min(len(mach_copy), idx+right+1)] = np.nan  # Exclude the forward shock region
    idx_reverse = np.nanargmax(mach_copy)
    r_reverse_shock = r[idx_reverse]

    if r_shock < r_reverse_shock:
        r_shock, r_reverse_shock = r_reverse_shock, r_shock  # Ensure r_shock is the larger radius (forward shock)
        
    return r_shock, r_reverse_shock

def _weighted_percentile(values, weights, percentiles):
    """Percentile(s) of `values` weighted by `weights` (linear interpolation)."""
    values  = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order   = np.argsort(values)
    values, weights = values[order], weights[order]
    # Cumulative weight at the midpoint of each cell's weight, normalised to [0, 100]
    cum = np.cumsum(weights) - 0.5 * weights
    cum /= np.sum(weights)
    return np.interp(np.asarray(percentiles) / 100.0, cum, values)

def find_shell_radius(s):
    # Select cells where the wind tracer is greater than the threshold
    unit_v = s.header['UnitVelocity_in_cm_per_s']
    mask = s.data['wind'] > 1e-4
    wind_r = s.data['r'][mask]

    # Handle the case where no cells match the criteria
    if len(wind_r) == 0:
        return np.nan, np.nan

    # Calculate the 95th (2σ) and 99.7th (3σ) percentiles of the radial distances
    # r_shell_l = np.percentile(wind_r, 95)
    # r_shell_u = np.percentile(wind_r, 99.7)

    # Mass-weight the percentiles so the heavily refined (many small cells)
    # inner region doesn't bias the radius inward. mass = rho * vol.
    wind_m = (s.data['rho'][mask] * s.data['vol'][mask])
    r_shell_l, r_shell_u = _weighted_percentile(wind_r, wind_m, [95, 99.7])

    return r_shell_l, r_shell_u

def find_Rcool(snappath, snapnum, L_AGN=1e45, v_w = 1e4):
    s = load_snap_data(snapnum, snappath=snappath, snapbase='snap_')
    s0 = load_snap_data(0, snappath=snappath, snapbase='snap_')

    unit_v = s.header['UnitVelocity_in_cm_per_s']

        # Normalisation constants
    eps0  = 0.05
    L0    = 1e45   # erg/s
    n0    = 1.0    # cm^-3
    t0    = 1.0    # Myr

    c = 3e5  # Speed of light in km/s
    epsilon = 0.5 * (v_w / c) 
    n_H = np.nanmedian(s0.data['nH_cm'])  # Pre-shock hydrogen number density in cm^-3, from initial snapshot
    t = calc_snap_time(s)  # Time in Myr
    t_s = t*u.Myr.to(u.s)  # Time in seconds

    # Define R in kpc
    Rdotsh = (3/5)*(epsilon / eps0)**(1/5) \
    * (L_AGN   / L0)**(1/5) \
    * (n_H     / n0)**(-1/5) \
    * (t       / t0)**(-2/5) # in kpc/Myr

    r_shell_l, r_shell_u = find_shell_radius(s)
    Rdotsh = (3/5)* r_shell_u / (t/t0)# In kpc, using the upper shell radius as a proxy for shock radius

    factor = (1 * u.kpc / u.Myr).to(u.km / u.s)
    Rdotsh_kms = Rdotsh * factor.value  # Convert to km/s

    # # Select cells where the cooling time is less than 1 Myr
    # mask = (s.data['wind'] > 0.5) & (s.data['vrad']*unit_v / 1e5  > 10)
    # Rdotsh = s.data['vrad'][mask]
    # Rdotsh_kms = np.nanmean(np.abs(Rdotsh))*unit_v / 1e5  # cm s^-1 -> km s^-1

    n0 = np.nanmedian(s0.data['n_dens_cm']/calc_mu(s0.data))  # Pre-shock density in cm^-3


    # Free-expansion cooling radius (R_c^ff) in kpc:
    r_cff_kpc = 3.8 * (Rdotsh_kms / 1e3)**2 * (n0)**(-1)

    print(f'R_c^ff = {r_cff_kpc:.3f} kpc  (Rdotsh = {Rdotsh_kms:.2f} km/s, n0 = {n0:.3e} cm^-3)')

    # r_cool_comp = 0.3 * (L_AGN / 1e45) * (Rdotsh_kms / 1e3)**(-1)
    # r_cool = r_cool_pc / 1e3  # kpc
    
    return r_cff_kpc, Rdotsh_kms, n0

def calculate_R_free(beta, tau, b, L_AGN, rho_0):
    """
    Calculate R_free (free expansion radius) based on equation (15).
    
    Parameters:
    -----------
    beta : float
        Beta parameter (dimensionless)
    tau : float
        Optical depth parameter (dimensionless)
    b : float
        Parameter b (dimensionless)
    L_AGN : float
        AGN luminosity in erg s^(-1)
    n_0 : float
        Number density in cm^(-3)
    
    Returns:
    --------
    R_free : float
        Free expansion radius in parsecs (pc)
    
    Formula:
    --------
    R_free ≈ 10 * (β / 0.1)^(-1) * (τ / b)^(1/2) * 
             (L_AGN / (10^45 erg s^(-1)))^(1/2) * 
             (n_0 / cm^(-3))^(-1/2) pc
    """
    c = 3e10  # speed of light in cm s^(-1)
    
    M_dot_W = (tau * L_AGN) / (beta * c**2)
    v_W = beta * c
    
    t_free = np.sqrt((3 / (4 * np.pi * b)) * (M_dot_W / (rho_0 * v_W**3)))
    R_free = v_W * t_free / (3.086e18)  # Convert cm to pc
    
    return R_free, t_free