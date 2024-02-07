import matplotlib.pyplot as plt

def arrange_phase_space_plot():

    plt.figure(figsize=(10, 5))
    ax_geom = plt.subplot(1, 2, 1, title='Physical phase space')
    ax_norm = plt.subplot(1, 2, 2, title='Normalized phase space')
    ax_geom.set_xlim(-5e-2, 5e-2); ax_geom.set_ylim(-5e-3, 5e-3)
    ax_norm.set_xlim(-15e-3, 15e-3); ax_norm.set_ylim(-15e-3, 15e-3)
    ax_norm.set_aspect('equal', adjustable='datalim')
    ax_norm.set_xlabel(r'$\hat{x}$')
    ax_norm.set_ylabel(r'$\hat{px}$')

    return ax_geom, ax_norm