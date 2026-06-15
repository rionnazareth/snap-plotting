
if __name__ == "__main__":
    from tests.lib import *
    import sys, os
    import cmasher
    sys.path.insert(0, '/cosma8/data/dp317/dc-naza3/arepo-snap-util')
    print('Running snap plotting script...')
    BASE_PATH = '/cosma8/data/dp317/dc-naza3/homogeneous'
    SNAPBASE = 'snap_'
    SNAPFILETYPE = '.hdf5'

    GAMMA = 5/3 # heat capaticy ratio for monatomic gas
    HYDROGENMASS_FRAC = 0.76

    k_B       = 1.381e-16
    m_p       = 1.66e-24

    ## Set up figure & axis grid
    plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        14,
    'axes.labelsize':   18,
    'axes.titlesize':   18,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize':  14,
    'ytick.labelsize':  14,
    'xtick.major.size': 7,
    'xtick.minor.size': 4,
    'ytick.major.size': 7,
    'ytick.minor.size': 4,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'axes.linewidth':   1.5,
    'legend.fontsize':  13,
    'legend.title_fontsize': 14,
        })
    fig = plt.figure(figsize=(10,10))

    outer = gridspec.GridSpec(1, 1, wspace=0.2)
    quad_subs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], hspace=0, wspace=0)

    ## Set global figure options

    image_proj = 'top'     # side or top viewing angle
    plotsize =  0.5   # Size in kpc of one panel
    proj_on = False    # Whether to do a slice or a projection
    proj_fact = 0.01         # Fraction of plotsize to project through
    res = 1024               # Pixels per panel

    ## Load the snapshot
    v = 'temp'
    c = 'viridis'
    r = [1e5,1e9]
    snap_num = 10
    output_path = BASE_PATH + '/rhov_hires/5/'

    slurm_ntasks = os.getenv('SLURM_NTASKS', '').strip()
    numthreads = int(slurm_ntasks) if slurm_ntasks.isdigit() and int(slurm_ntasks) > 0 else 1

    s = load_snap_data(snap_num,snappath=output_path,snapbase=SNAPBASE)

    unit_v = s.header['UnitVelocity_in_cm_per_s']
    unit_l = s.header['UnitLength_in_cm'] 
    unit_m = s.header['UnitMass_in_g']
    unit_t = unit_l / unit_v
    unit_rho = unit_m / unit_l**3
    unit_pres = unit_rho * (unit_v**2)

    s0 = load_snap_data(0,snappath=output_path,snapbase=SNAPBASE)
    norm = True
    if norm:
        for data_key in ['vrad', 'cren']:
            # Avoid division by zero and cast to float to avoid UFuncTypeError
            div = s0.data[data_key].mean() if s0.data[data_key].mean() != 0 else 1e-10
            if data_key == 'cren': div = np.median(s0.data['u'])  # Normalize CR energy by initial internal energy, not CR energy:
            if data_key == 'vrad': div = 1e5/unit_v  # Normalize velocities by 1000 km/s to get more manageable numbers
            s.data[data_key] = s.data[data_key].astype(float) / div
            print(f'Normalized {data_key} by dividing by max value from snap 0: {s0.data[data_key].mean()}')
            
    snap_time = calc_snap_time(s)
    add_vec = False
    ## Plot each axis quadrent
    # Top left
    norm_by_snap0(norm)
    quad_TL, map_TL = plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [0,0],
        var = 'temp',
        weighted = 'rho', # or None
        ranges = [5e4,1e9],
        cmap = 'gnuplot',
        logplot = True,
        divzero = False,
        divzero_centre = None,
        colorbar=False,
        image_proj = image_proj,
        proj_on = proj_on,
        proj_fact = proj_fact,
        res = res,
        plotsize = plotsize,
        add_vec=add_vec,
        vec_val='bfld',
        numthreads=numthreads
        )

    # output_path = BASE_PATH + '/output_lb/'

    # s = load_snap_data(snap_num,snappath=output_path,snapbase=SNAPBASE)
    # snap_time = calc_snap_time(s)
    # add_vec = True
    # Bottom left

    quad_BL, map_BL = plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [1,0],
        var = 'nH_cm',
        weighted = 'rho', # or None
        ranges = [1e-3,1e2],
        cmap = 'jet',
        logplot = True,
        divzero = False,
        divzero_centre = None,
        colorbar=False,
        image_proj = image_proj,
        proj_on = proj_on,
        proj_fact = proj_fact,
        res = res,
        plotsize = plotsize,
        add_vec=add_vec,
        vec_val='bfld',
        numthreads=numthreads
        )

    # Bottom right

    quad_BR, map_BR = plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [1,1],
        var = 'vrad',
        weighted = 'rho', # or None
        ranges = [2e1,1e4],
        cmap = 'cmr.ember',
        logplot = True,
        divzero = False,
        divzero_centre = None,
        colorbar=False,
        image_proj = image_proj,
        proj_on = proj_on,
        proj_fact = proj_fact,
        res = res,
        plotsize = plotsize,
        add_vec=add_vec,
        vec_val='bfld',
        numthreads=numthreads
        )

    # Top right
    # output_path = BASE_PATH + '/output_obli/'

    # s = load_snap_data(snap_num,snappath=output_path,snapbase=SNAPBASE)
    # snap_time = calc_snap_time(s)

    quad_TR, map_TR = plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [0,1],
        var = 'crendens',
        weighted = 'rho', # or None
        ranges = [2e-2,1e1],
        cmap = 'cmr.amethyst',
        logplot = True,
        divzero = False,
        divzero_centre = None,
        colorbar=False,
        image_proj = image_proj,
        proj_on = proj_on,
        proj_fact = proj_fact,
        res = res,
        plotsize = plotsize,
        add_vec=add_vec,
        vec_val='bfld',
        numthreads=numthreads
        )
    for ax in [quad_TL, quad_TR, quad_BL, quad_BR]:
        ax.set_xticks([])
        ax.set_yticks([])
    # r_shock, r_reverse = find_shock_radius(s, r_range=(1e-3,plotsize), nbins=500)

    revshock = True
    shock = True
    if shock:
        for ax in [quad_TL, quad_TR, quad_BL, quad_BR]:
            ax.add_patch(plt.Circle((s.header['BoxSize']/2., s.header['BoxSize']/2.), r_shock, color='yellow', fill=False, linestyle='--', linewidth=1.5, label='Forward Shock'))
            # ax.add_patch(plt.Circle((s_nocr.header['BoxSize']/2., s_nocr.header['BoxSize']/2.), r_nocr, color='magenta', fill=False, linestyle=':', linewidth=1.5, label='No-CR Shock'))
            if revshock:
                ax.add_patch(plt.Circle((s.header['BoxSize']/2., s.header['BoxSize']/2.), r_reverse, color='cyan', fill=False, linestyle='--', linewidth=1.5, label='Reverse Shock'))


    # plt.show()

    # fig.subplots_adjust(bottom=0.14, top=0.88)

    # Add colorbars - top row at the top, bottom row at the bottom
    cax_TL = fig.add_axes([quad_TL.get_position().x0, quad_TL.get_position().y1, 
                            quad_TL.get_position().width, 0.02])
    fig.colorbar(map_TL, cax=cax_TL, orientation='horizontal', ticklocation='top')#, label=r'$T/T_0$')

    cax_BL = fig.add_axes([quad_BL.get_position().x0, quad_BL.get_position().y0 - 0.02, 
                            quad_BL.get_position().width, 0.02])
    fig.colorbar(map_BL, cax=cax_BL, orientation='horizontal')#, label=r'$n_\mathrm{H}/n_0$')

    cax_TR = fig.add_axes([quad_TR.get_position().x0, quad_TR.get_position().y1, 
                            quad_TR.get_position().width, 0.02])
    cb_TR = fig.colorbar(map_TR, cax=cax_TR, orientation='horizontal', ticklocation='top')#, label=r'$E_\mathrm{CR}/E_0$')
    # cb_TR.set_ticks(cb_TR.get_ticks()[1:])

    cax_BR = fig.add_axes([quad_BR.get_position().x0, quad_BR.get_position().y0 - 0.02, 
                            quad_BR.get_position().width, 0.02])

    cb_BR = fig.colorbar(map_BR, cax=cax_BR, orientation='horizontal')#, label=r'$X_\mathrm{CR}=P_\mathrm{CR}/P_{th}$')
    # cb_BR.set_ticks(cb_BR.get_ticks()[1:])

    quad_TL.text(0.05, 0.95, r'$T$ [K]', transform=quad_TL.transAxes, color='white', ha='left', va='top', weight='bold',fontsize=20)
    quad_BL.text(0.05, 0.05, r'$n_\mathrm{H}$ [cm$^{-3}$]', transform=quad_BL.transAxes, color='black', ha='left', va='bottom', weight='bold',fontsize=20)
    quad_TR.text(0.95, 0.95, r'$\epsilon_\mathrm{CR}$', transform=quad_TR.transAxes, color='white', ha='right', va='top', weight='bold',fontsize=20)
    quad_BR.text(0.95, 0.05, r'$v_\mathrm{rad}$ [km/s]', transform=quad_BR.transAxes, color='white', ha='right', va='bottom', weight='bold',fontsize=20)

    # fig.suptitle(f'Snapshot {snap_num:03d} — Time: {snap_time:.1f} Myr — {v}', fontsize=12, y=1.002)
    inset = False
    if inset:
        # Add an inset to the bottom right quadrant
        y_idx = 2 if image_proj == 'side' else 1

        center_zoom = [0.6, 0.5, 0.06] # x, y, z
        box_zoom = [0.05, 0.05]
        
        axins = quad_BR.inset_axes([center_zoom[0] - box_zoom[0]/2.+0.25, center_zoom[y_idx] - box_zoom[1]/2.+0.15, 0.4, 0.4])

        s.axplot_Aweightedslice(axins,
                        value='xcr', weights='rho', cmap='Blues', colorbar=False, logplot=True, vrange=[2e-4,1e3],
                        center=center_zoom, box=box_zoom, res=res,
                        proj=proj_on, proj_fact=proj_fact, numthreads=numthreads, axes=[0,2] if image_proj == 'side' else [0,1]
                    )
        
        # Set the limits of the inset axes to match the zoom box
        axins.set_xlim(center_zoom[0] - box_zoom[0]/2., center_zoom[0] + box_zoom[0]/2.)
        # Depending on image_proj, the y-axis of the plot corresponds to either y or z
        axins.set_ylim(center_zoom[y_idx] - box_zoom[1]/2., center_zoom[y_idx] + box_zoom[1]/2.)
        
        # Add shocks circle to the inset
        axins.add_patch(plt.Circle((s.header['BoxSize']/2., s.header['BoxSize']/2.), r_shock, color='magenta', fill=False, linestyle='--', linewidth=1.5))
        axins.add_patch(plt.Circle((s.header['BoxSize']/2., s.header['BoxSize']/2.), r_nocr, color='maroon', fill=False, linestyle=':', linewidth=1.5))

        if revshock:
            axins.add_patch(plt.Circle((s.header['BoxSize']/2., s.header['BoxSize']/2.), r_reverse, color='cyan', fill=False, linestyle='--', linewidth=1.5))


        axins.set_xticks([])
        axins.set_yticks([])
        # axins.autoscale(False)
        quad_BR.indicate_inset_zoom(axins, edgecolor="gray")

    fig.savefig('/cosma8/data/dp317/dc-naza3/snap-plotting/tests/crph/{}_snap{}_{}.png'.format(v,number_string(snap_num),image_proj),dpi=300)
    # plt.show()