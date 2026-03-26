if __name__ == "__main__":
    from tests.lib import *
    print('Running snap plotting script...')
    BASE_PATH = '/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion'
    SNAPBASE = 'snap_'
    SNAPFILETYPE = '.hdf5'

    GAMMA = 5/3 # heat capaticy ratio for monatomic gas
    HYDROGENMASS_FRAC = 0.76

    k_B       = 1.381e-16
    m_p       = 1.66e-24

    ## Set up figure & axis grid
    fig = plt.figure(figsize=(8,8))

    outer = gridspec.GridSpec(1, 1, wspace=0.2)
    quad_subs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], hspace=0, wspace=0)

    ## Set global figure options

    image_proj = 'side'     # side or top viewing angle
    plotsize =  0.5   # Size in kpc of one panel
    proj_on = False     # Whether to do a slice or a projection
    proj_fact = 0.01         # Fraction of plotsize to project through
    res = 1024               # Pixels per panel

    ## Load the snapshot
    v = 'temp'
    c = 'viridis'
    r = [1e5,1e9]
    snap_num = 10
    output_path = BASE_PATH + '/old/output_cr600/'

    s = load_snap_data(snap_num,snappath=output_path,snapbase=SNAPBASE)

    s0 = load_snap_data(0,snappath=output_path,snapbase=SNAPBASE)
    norm = True
    if norm:
        for data_key in ['temp', 'nH_cm', 'cren']:
            # Avoid division by zero and cast to float to avoid UFuncTypeError
            div = s0.data[data_key].mean() if s0.data[data_key].mean() != 0 else 1e-10
            if data_key == 'cren': div = s0.data['u'].mean()  # Normalize CR energy by initial internal energy, not CR energy:
            s.data[data_key] = s.data[data_key].astype(float) / div
            print(f'Normalized {data_key} by dividing by max value from snap 0: {s0.data[data_key].mean()}')
            
    snap_time = calc_snap_time(s)
    add_vec = False
    ## Plot each axis quadrent
    # Top left
    quad_TL, map_TL = plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [0,0],
        var = 'temp',
        weighted = 'rho', # or None
        ranges = [1,1e5],
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
        vec_val='bfld'
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
        ranges = [1e-3,1e1],
        cmap = 'viridis',
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
        vec_val='bfld'
        )

    # Bottom right

    quad_BR, map_BR = plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [1,1],
        var = 'xcr',
        weighted = 'rho', # or None
        ranges = [2e-4,1e3],
        cmap = 'Blues',
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
        vec_val='bfld'
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
        var = 'cren',
        weighted = 'rho', # or None
        ranges = [5e-5,1e4],
        cmap = 'cividis',
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
        vec_val='bfld'
        )
    for ax in [quad_TL, quad_TR, quad_BL, quad_BR]:
        ax.set_xticks([])
        ax.set_yticks([])
    r_shock, r_reverse = find_shock_radius(s, r_range=(1e-3,plotsize), nbins=500)

    s_nocr = load_snap_data(snap_num,snappath=BASE_PATH + '/old/output_homo/',snapbase=SNAPBASE)
    r_nocr, r_revnocr= find_shock_radius(s_nocr, r_range=(1e-3,plotsize), nbins=500)
    revshock = False
    for ax in [quad_TL, quad_TR, quad_BL, quad_BR]:
        ax.add_patch(plt.Circle((s.header['BoxSize']/2., s.header['BoxSize']/2.), r_shock, color='magenta', fill=False, linestyle='--', linewidth=1.5, label='Forward Shock'))
        # ax.add_patch(plt.Circle((s_nocr.header['BoxSize']/2., s_nocr.header['BoxSize']/2.), r_nocr, color='magenta', fill=False, linestyle=':', linewidth=1.5, label='No-CR Shock'))
        if revshock:
            ax.add_patch(plt.Circle((s.header['BoxSize']/2., s.header['BoxSize']/2.), r_reverse, color='cyan', fill=False, linestyle='--', linewidth=1.5, label='Reverse Shock'))
            ax.add_patch(plt.Circle((s_nocr.header['BoxSize']/2., s_nocr.header['BoxSize']/2.), r_revnocr, color='cyan', fill=False, linestyle=':', linewidth=1.5, label='No-CR Reverse Shock'))

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

    quad_TL.text(0.05, 0.95, r'$T/T_0$', transform=quad_TL.transAxes, color='white', ha='left', va='top', weight='bold',fontsize=12)
    quad_BL.text(0.05, 0.05, r'$n_\mathrm{H}/n_0$', transform=quad_BL.transAxes, color='black', ha='left', va='bottom', weight='bold',fontsize=12)
    quad_TR.text(0.95, 0.95, r'$E_\mathrm{CR}/E_0$', transform=quad_TR.transAxes, color='white', ha='right', va='top', weight='bold',fontsize=12)
    quad_BR.text(0.95, 0.05, r'$X_\mathrm{CR}=P_\mathrm{CR}/P_{th}$', transform=quad_BR.transAxes, color='white', ha='right', va='bottom', weight='bold',fontsize=12)
   
    # fig.suptitle(f'Snapshot {snap_num:03d} — Time: {snap_time:.1f} Myr — {v}', fontsize=12, y=1.002)
    fig.savefig('/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/snap-plotting/tests/pap/{}_snap{}_{}.pdf'.format(v,number_string(snap_num),image_proj),dpi=300)
    # plt.show()
    # s.plot_Aweightedslice(
    #                 value='xcr', weights='rho', cmap='Blues', colorbar=False, logplot=True, vrange=[2e-4,1e3],
    #                 center=[0.32, 0.32, 0.32], box=[0.1,0.1], res=res,
    #                 newfig=False, proj=proj_on, proj_fact=proj_fact
    #             )