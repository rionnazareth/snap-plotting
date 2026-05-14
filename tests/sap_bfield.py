if __name__ == "__main__":
    from lib import *
    import scienceplots
    plt.style.use('science')
    print('Running snap plotting script...')
    BASE_PATH = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion'
    SNAPBASE = 'snap_'
    SNAPFILETYPE = '.hdf5'

    GAMMA = 5/3 # heat capaticy ratio for monatomic gas
    HYDROGENMASS_FRAC = 0.76

    k_B       = 1.381e-16
    m_p       = 1.66e-24

    ## Set up figure & axis grid
    fig = plt.figure(figsize=(10,10))

    outer = gridspec.GridSpec(1, 1, wspace=0.2)
    quad_subs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], hspace=0, wspace=0)

    ## Set global figure options

    image_proj = 'side'     # side or top viewing angle
    plotsize =  0.5   # Size in kpc of one panel
    proj_on = False   # Whether to do a slice or a projection
    proj_fact = 0.1         # Fraction of plotsize to project through
    res = 1024               # Pixels per panel

    ## Load the snapshot
    v = 'crpres'
    c = 'vanimo' 
    r = None#[5e-6,1e-4]

    snap_num = 8
    output_path = BASE_PATH + '/new/output_cbcr/'

    slurm_ntasks = os.getenv('SLURM_NTASKS', '').strip()
    numthreads = int(slurm_ntasks) if slurm_ntasks.isdigit() and int(slurm_ntasks) > 0 else 1


    s = load_snap_data(snap_num,snappath=output_path,snapbase=SNAPBASE)
    snap_time = calc_snap_time(s)
    add_vec = False
    vec_val = 'bfld'
    logplot = True

    unit_v = s.header['UnitVelocity_in_cm_per_s']
    unit_l = s.header['UnitLength_in_cm'] 
    unit_m = s.header['UnitMass_in_g']
    unit_t = unit_l / unit_v
    unit_rho = unit_m / unit_l**3
    unit_b = np.sqrt(unit_rho * unit_v**2)

    norm = True
    cdis = False
    def norm_by_snap0(norm):
        s0 = load_snap_data(0,snappath=output_path,snapbase=SNAPBASE)
        if norm:
            for data_key in ['rho', 'nH_cm', 'pres',  'speed','temp', 'wind','vrad','bflds']:#
                # Avoid division by zero and cast to float to avoid UFuncTypeError
                div = np.median(s0.data[data_key]) if s0.data[data_key].mean() != 0 else 1e-10
                if data_key == 'cren': div = np.median(s0.data['u'])  # Normalize CR energy by initial internal energy, not CR energy:
                if data_key == 'crpres': div = np.median(s0.data['pres'])  # Normalize CR pressure by initial thermal pressure, not CR pressure
                if data_key == 'speed' or data_key == 'vrad': div = 1e5/unit_v # speed in kms
                if data_key == 'bflds': div = 1/unit_b # Convert magnetic field to physical units for normalization
                s.data[data_key] = s.data[data_key].astype(float) / div
                print(f'Normalized {data_key} by dividing by max value from snap 0: {np.median(s0.data[data_key])}')
        if cdis: 
            s.data[v] *= (s.data['wind']>=0.5)


    ## Plot each axis quadrent
    # Top left

    # Here we're passing the same snap each time, but you could give each one a different snapshot to make e.g. a time series image
    norm_by_snap0(norm)
    quad_TL, map_TL = plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [0,0],
        var = v,
        weighted = 'rho', # or None
        ranges = r,
        cmap = c,
        logplot = logplot,
        divzero = False,
        divzero_centre = None,
        colorbar=False,
        image_proj = image_proj,
        proj_on = proj_on,
        proj_fact = proj_fact,
        res = res,
        numthreads = numthreads,
        plotsize = plotsize,
        add_vec=add_vec,
        vec_val=vec_val
        )

    # output_path = BASE_PATH + '/output_cnob/'

    s = load_snap_data(snap_num,snappath=output_path,snapbase=SNAPBASE)
    snap_time = calc_snap_time(s)
    # Bottom left
    norm_by_snap0(norm)
    quad_BL, map_BL = plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [1,0],
        var = v,
        weighted = 'rho', # or None
        ranges = r,
        cmap = c,
        logplot = logplot,
        divzero = False,
        divzero_centre = None,
        colorbar=False,
        image_proj = image_proj,
        proj_on = proj_on,
        proj_fact = proj_fact,
        res = res,
        numthreads = numthreads,
        plotsize = plotsize,
        add_vec=add_vec,
        vec_val=vec_val
        )

    # snap_num = 1
    output_path = BASE_PATH + '/new/output_cbcr/'

    s = load_snap_data(snap_num,snappath=output_path,snapbase=SNAPBASE)
    snap_time = calc_snap_time(s)
    # Bottom right
    norm_by_snap0(norm)
    quad_BR, map_BR = plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [1,1],
        var = v,
        weighted = 'rho', # or None
        ranges = r,
        cmap = c,
        logplot = logplot,
        divzero = False,
        divzero_centre = None,
        colorbar=False,
        image_proj = image_proj,
        proj_on = proj_on,
        proj_fact = proj_fact,
        res = res,
        numthreads = numthreads,
        plotsize = plotsize,
        add_vec=add_vec,
        vec_val=vec_val
        )

    # Top right
    # output_path = BASE_PATH + '/output_cturb/'

    s = load_snap_data(snap_num,snappath=output_path,snapbase=SNAPBASE)
    snap_time = calc_snap_time(s)
    norm_by_snap0(norm)
    quad_TR, map_TR = plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [0,1],
        var = v,
        weighted = 'rho', # or None
        ranges = r,
        cmap = c,
        logplot = logplot,
        divzero = False,
        divzero_centre = None,
        colorbar=False,
        image_proj = image_proj,
        proj_on = proj_on,
        proj_fact = proj_fact,
        res = res,
        numthreads = numthreads,
        plotsize = plotsize,
        add_vec=add_vec,
        vec_val=vec_val
        )
    # for ax in [quad_TL, quad_TR, quad_BL, quad_BR]:
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    r_shock, r_shocku = find_shock_radius(s, r_range=(1e-3,plotsize), nbins=500)#find_shell_radius(s)#

    s_nocr = load_snap_data(snap_num,snappath=BASE_PATH + '/old/output_homo/',snapbase=SNAPBASE)
    r_nocr, r_nocru= find_shock_radius(s_nocr, r_range=(1e-3,plotsize), nbins=500)#find_shell_radius(s_nocr)#
    s_nodiff = load_snap_data(snap_num,snappath=BASE_PATH + '/old/output_cr600/',snapbase=SNAPBASE)
    r_nodiff, r_nodiffu= find_shock_radius(s_nodiff, r_range=(1e-3,plotsize), nbins=500)#find_shell_radius(s_nodiff)

    revshock = False
    # for ax in [quad_TL, quad_TR, quad_BL, quad_BR]:
    #     ax.add_patch(plt.Circle((s.header['BoxSize']/2., s.header['BoxSize']/2.), r_shocku, color='white', fill=False, linestyle='--', linewidth=1.5, label='Forward Shock'))
        # ax.add_patch(plt.Circle((s_nocr.header['BoxSize']/2., s_nocr.header['BoxSize']/2.), r_nocr, color='magenta', fill=False, linestyle=':', linewidth=1.5, label='No-CR Shock'))
        # if revshock:
        #     ax.add_patch(plt.Circle((s.header['BoxSize']/2., s.header['BoxSize']/2.), r_reverse, color='cyan', fill=False, linestyle='--', linewidth=1.5, label='Reverse Shock'))
        #     ax.add_patch(plt.Circle((s_nocr.header['BoxSize']/2., s_nocr.header['BoxSize']/2.), r_revnocr, color='cyan', fill=False, linestyle=':', linewidth=1.5, label='No-CR Reverse Shock'))

    # plt.show()

    # fig.subplots_adjust(bottom=0.14, top=0.88)

    # Add colorbars - top row at the top, bottom row at the bottom
    # cax_TL = fig.add_axes([quad_TL.get_position().x0, quad_TL.get_position().y1, 
    #                        quad_TL.get_position().width, 0.02])
    # fig.colorbar(map_TL, cax=cax_TL, orientation='horizontal', ticklocation='top')#, label=r'$T/T_0$')
    
    # cax_BL = fig.add_axes([quad_BL.get_position().x0, quad_BL.get_position().y0 - 0.02, 
    #                        quad_BL.get_position().width, 0.02])
    # fig.colorbar(map_BL, cax=cax_BL, orientation='horizontal')#, label=r'$n_\mathrm{H}/n_0$')
    
    # cax_TR = fig.add_axes([quad_TR.get_position().x0, quad_TR.get_position().y1, 
    #                        quad_TR.get_position().width, 0.02])
    # cb_TR = fig.colorbar(map_TR, cax=cax_TR, orientation='horizontal', ticklocation='top')#, label=r'$E_\mathrm{CR}/E_0$')
    # # cb_TR.set_ticks(cb_TR.get_ticks()[1:])
    
    # cax_BR = fig.add_axes([quad_BR.get_position().x0, quad_BR.get_position().y0 - 0.02, 
    #                        quad_BR.get_position().width, 0.02])
    
    # cb_BR = fig.colorbar(map_BR, cax=cax_BR, orientation='horizontal')#, label=r'$X_\mathrm{CR}=P_\mathrm{CR}/P_{th}$')
    # cb_BR.set_ticks(cb_BR.get_ticks()[1:])

    # quad_TL.text(0.05, 0.95, r'$T/T_0$', transform=quad_TL.transAxes, color='white', ha='left', va='top', weight='bold',fontsize=15)
    quad_TL.text(0.05, 0.95, r'without CR', transform=quad_TL.transAxes, color='white', ha='left', va='top', weight='bold',fontsize=20)

    # quad_BL.text(0.05, 0.05, r'$B = 0 \; \mathrm{G}$+diff', transform=quad_BL.transAxes, color='black', ha='left', va='bottom', weight='bold',fontsize=15)
    quad_TR.text(0.95, 0.95, r'with CR', transform=quad_TR.transAxes, color='white', ha='right', va='top', weight='bold',fontsize=20)
    # quad_BR.text(0.95, 0.05, r'$B = 0 \; \mathrm{G}$+no diff', transform=quad_BR.transAxes, color='black', ha='right', va='bottom', weight='bold',fontsize=15)
   
    # fig.suptitle(r'$|\vec{B}|$', fontsize=18, y=1.002)
    inset = False
    if inset:
        # Add an inset to the bottom right quadrant
        y_idx = 2 if image_proj == 'side' else 1

        center_zoom = [0.6, 0.5, 0.06] # x, y, z
        box_zoom = [0.1, 0.1]#increase to make the zoom box bigger, decrease to make it smaller
        
        axins = quad_BR.inset_axes([center_zoom[0] - box_zoom[0]/2.+0.25, center_zoom[y_idx] - box_zoom[1]/2.+0.15, 0.4, 0.4])

        s.axplot_Aweightedslice(axins,
                        value=v, weights='rho', cmap='jet', colorbar=False, logplot=True, vrange=r,
                        center=center_zoom, box=box_zoom, res=res,
                        proj=proj_on, proj_fact=proj_fact, numthreads=numthreads, axes=[0,2] if image_proj == 'side' else [0,1]
                    )
        
        # Set the limits of the inset axes to match the zoom box
        axins.set_xlim(center_zoom[0] - box_zoom[0]/2., center_zoom[0] + box_zoom[0]/2.)
        # Depending on image_proj, the y-axis of the plot corresponds to either y or z
        axins.set_ylim(center_zoom[y_idx] - box_zoom[1]/2., center_zoom[y_idx] + box_zoom[1]/2.)
        
        # Add shocks circle to the inset
        axins.add_patch(plt.Circle((s.header['BoxSize']/2., s.header['BoxSize']/2.), r_shocku, color='white', fill=False, linestyle='--', linewidth=1.5))
        axins.add_patch(plt.Circle((s.header['BoxSize']/2., s.header['BoxSize']/2.), r_nocr, color='cyan', fill=False, linestyle=':', linewidth=1.5))
        axins.add_patch(plt.Circle((s.header['BoxSize']/2., s.header['BoxSize']/2.), r_nodiff, color='red', fill=False, linestyle=':', linewidth=1.5))

        # if revshock:
        #     axins.add_patch(plt.Circle((s.header['BoxSize']/2., s.header['BoxSize']/2.), r_reverse, color='cyan', fill=False, linestyle='--', linewidth=1.5))


        axins.set_xticks([])
        axins.set_yticks([])
        # axins.autoscale(False)
        quad_BR.indicate_inset_zoom(axins, edgecolor="gray")

    # Add a single large colorbar spanning the full height on the right side
    cax = fig.add_axes([
        quad_TR.get_position().x1 + 0.02,                      # x position: slightly right of the right col
        quad_BR.get_position().y0,                             # y position: bottom edge of bottom plots
        0.03,                                                  # width: 3% of figure width
        quad_TR.get_position().y1 - quad_BR.get_position().y0  # height: from bottom of BR to top of TR
    ])
    fig.colorbar(map_TR, cax=cax, orientation='vertical', label=r'$|\vec{B}| \; \mathrm{[G]}$')

    fig.savefig('/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/snap-plotting/tests/bfield_amp/{}_snap{}_{}.png'.format(v,number_string(snap_num),image_proj),dpi=300)
    # plt.show()