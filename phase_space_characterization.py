import numpy as np
import matplotlib.pyplot as plt

def characterize_phase_space_at_septum(line, num_turns=1000, plot=False):

    tw = line.twiss(method='4d')

    # Localize transition between stable and unstable
    x_septum = 3.5e-2

    x_stable = 0
    x_unstable = 3e-2
    while x_unstable - x_stable > 1e-6:
        x_test = (x_stable + x_unstable) / 2
        p = line.build_particles(x=x_test, px=0)
        line.track(p, num_turns=num_turns, turn_by_turn_monitor=True)
        mon_test = line.record_last_track
        if (mon_test.x > x_septum).any():
            x_unstable = x_test
        else:
            x_stable = x_test

    p = line.build_particles(x=[x_stable, x_unstable], px=0)
    line.track(p, num_turns=num_turns, turn_by_turn_monitor=True)
    mon_separatrix = line.record_last_track
    nc_sep = tw.get_normalized_coordinates(mon_separatrix)

    z_triang = nc_sep.x_norm[0, :] + 1j * nc_sep.px_norm[0, :]
    r_triang = np.abs(z_triang)

    # Find fixed points
    i_fp1 = np.argmax(r_triang)
    z_fp1 = z_triang[i_fp1]
    r_fp1 = np.abs(z_fp1)

    mask_fp2 = np.abs(z_triang - z_fp1 * np.exp(1j * 2 / 3 * np.pi)) < 0.2 * r_fp1
    i_fp2 = np.argmax(r_triang * mask_fp2)

    mask_fp3 = np.abs(z_triang - z_fp1 * np.exp(-1j * 2 / 3 * np.pi)) < 0.2 * r_fp1
    i_fp3 = np.argmax(r_triang * mask_fp3)

    x_norm_fp = np.array([nc_sep.x_norm[0, i_fp1],
                          nc_sep.x_norm[0, i_fp2],
                          nc_sep.x_norm[0, i_fp3]])
    px_norm_fp = np.array([nc_sep.px_norm[0, i_fp1],
                           nc_sep.px_norm[0, i_fp2],
                           nc_sep.px_norm[0, i_fp3]])

    x_fp = np.array([mon_separatrix.x[0, i_fp1],
                     mon_separatrix.x[0, i_fp2],
                     mon_separatrix.x[0, i_fp3]])
    px_fp = np.array([mon_separatrix.px[0, i_fp1],
                      mon_separatrix.px[0, i_fp2],
                      mon_separatrix.px[0, i_fp3]])

    stable_area = np.linalg.det([x_norm_fp, px_norm_fp, [1, 1, 1]])

    # Measure slope of the separatrix at the semptum
    x_separ = mon_separatrix.x[1, :].copy()
    px_separ = mon_separatrix.px[1, :].copy()
    x_norm_separ = nc_sep.x_norm[1, :].copy()
    px_norm_separ = nc_sep.px_norm[1, :].copy()

    x_separ[px_norm_separ < -1e-2] = 99999999. # Mask away second separatrix

    i_septum = np.argmin(np.abs(x_separ - x_septum))

    poly_sep = np.polyfit([x_separ[i_septum + 3], x_separ[i_septum - 3]],
                             [px_separ[i_septum + 3], px_separ[i_septum - 3]],
                              deg=1)
    dpx_dx_at_septum = poly_sep[0]

    if plot:
        x = np.linspace(0, 1.2*x_stable, 15)
        particles = line.build_particles(x=x, px=0)
        line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True)
        mon = line.record_last_track
        nc = tw.get_normalized_coordinates(mon)

        plt.figure(figsize=(10, 5))
        ax_geom = plt.subplot(1, 2, 1)
        plt.plot(mon.x.T, mon.px.T, '.', markersize=1, color='C0')
        plt.ylabel(r'$p_x$')
        plt.xlabel(r'$x$ [m]')
        plt.xlim(-5e-2, 5e-2)
        plt.ylim(-5e-3, 5e-3)
        ax_norm = plt.subplot(1, 2, 2)
        plt.plot(nc.x_norm.T * 1e3, nc.px_norm.T * 1e3,
                 '.', markersize=1, color='C0')
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        plt.gca().set_aspect('equal', adjustable='datalim')

        plt.xlabel(r'$\hat{x}$ [$10^{-3}$]')
        plt.ylabel(r'$\hat{y}$ [$10^{-3}$]')

        # Plot separatrix
        x_triang =mon_separatrix.x[0, :]
        px_triang = mon_separatrix.px[0, :]
        x_norm_triang = nc_sep.x_norm[0, :]
        px_norm_triang = nc_sep.px_norm[0, :]

        theta_triang = np.angle(x_norm_triang + 1j * px_norm_triang)
        idx = np.argsort(theta_triang)
        x_triang = x_triang[idx]
        px_triang = px_triang[idx]
        x_norm_triang = x_norm_triang[idx]
        px_norm_triang = px_norm_triang[idx]

        mask_alive = mon_separatrix.state[1, :] > 0
        for ii in range(3):
            ax_geom.plot(mon_separatrix.x[1, mask_alive][ii::3],
                         mon_separatrix.px[1, mask_alive][ii::3],
                         '-', lw=3, color='C1', alpha=0.9)
        ax_geom.plot(x_triang, px_triang, '-', lw=3, color='C2', alpha=0.9)
        ax_geom.plot(x_fp, px_fp, '*', markersize=10, color='k')

        for ii in range(3):
            ax_norm.plot(nc_sep.x_norm[1, mask_alive][ii::3] * 1e3,
                         nc_sep.px_norm[1, mask_alive][ii::3] * 1e3,
                         '-', lw=3, color='C1', alpha=0.9)
        ax_norm.plot(x_norm_triang * 1e3, px_norm_triang * 1e3,
                     '-', lw=3, color='C2', alpha=0.9)
        ax_norm.plot(x_norm_fp*1e3, px_norm_fp*1e3, '*', markersize=10, color='k')

        x_plt = [x_septum - 1e-2, x_septum + 1e-2]
        ax_geom.plot(x_plt, np.polyval(poly_sep, x_plt), '--k', linewidth=3)
        ax_geom.axvline(x=x_septum, color='k', alpha=0.4, linestyle='--')
        plt.subplots_adjust(wspace=0.3)
        ax_geom.set_title('Physical phase space')
        ax_norm.set_title('Normalized phase space')

    return {
        'dpx_dx_at_septum': dpx_dx_at_septum,
        'stable_area': stable_area,
        'x_fp': x_fp,
        'px_fp': x_fp,
        'x_norm_fp': x_norm_fp,
        'px_norm_fp': x_norm_fp,
    }