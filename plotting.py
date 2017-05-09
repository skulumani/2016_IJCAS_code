"""Plot the outputs of the simulation

"""
import numpy as np

import matplotlib 
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import colors, animation, cm

# parse out the simulation results and recompute the error functions for 

def figsize(scale):
    fig_width_pt = 483.0                            # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def scale_figsize(wscale, hscale):
    fig_width_pt = 483.0
    fig_height_pt = 682.0
    inches_per_pt = 1 / 72.27
    fig_width = fig_width_pt * inches_per_pt * wscale
    fig_height = fig_height_pt * inches_per_pt * hscale
    return (fig_width, fig_height)

# PGF with LaTeX
pgf_with_latex = {                      # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 8,               # LaTeX default is 10pt font.
        "font.size": 8,
        "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
        "figure.autolayout": True,
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
            r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
            r"\usepackage{siunitx}",
            ]
        }

matplotlib.rcParams.update(pgf_with_latex)
sns.set_style('whitegrid', pgf_with_latex)
sns.color_palette('bright')
time_label = r'$t$ (sec)'
linewidth=1
def plot_outputs(sc, fname_suffix='', wscale=1, hscale=0.75, pgf_save=False):
    """Given a simulated rigid body instantiation this will plot all the outputs
    
    Parameters:
    ----------
    time: (n,) numpy array
    sc: rigid-body object

    """
    # plot eR components
    er_fig, er_axarr = plt.subplots(3,1, sharex=True, figsize=scale_figsize(wscale, hscale))
    er_axarr[0].plot(sc.time, sc.err_att[:,0], linewidth=linewidth, label='Actual',
                      linestyle='-')
    er_axarr[0].plot(sc.time, np.zeros_like(sc.err_att[:,0]), linewidth=linewidth, label='Desired',
                     linestyle='--')
    er_axarr[0].set_ylabel(r'$e_{R_1}$')
    er_axarr[1].plot(sc.time, sc.err_att[:,1], linewidth=linewidth, label='Actual',
                      linestyle='-')
    er_axarr[1].plot(sc.time, np.zeros_like(sc.err_att[:,1]), linewidth=linewidth, label='Desired',
                     linestyle='--')
    er_axarr[1].set_ylabel(r'$e_{R_2}$')
    er_axarr[2].plot(sc.time, sc.err_att[:,2], linewidth=linewidth, label='Actual',
                      linestyle='-')
    er_axarr[2].plot(sc.time, np.zeros_like(sc.err_att[:,2]), linewidth=linewidth, label='Desired',
                     linestyle='--')
    er_axarr[2].set_ylabel(r'$e_{R_3}$')
    er_axarr[2].set_xlabel(time_label)
    plt.tight_layout()

    # plot configuration error function \Psi
    psi_fig, psi_ax = plt.subplots(1, 1, figsize=scale_figsize(wscale, hscale))
    psi_ax.plot(sc.time, sc.Psi, linewidth=linewidth, linestyle='-', label='Actual')
    psi_ax.plot(sc.time, np.zeros_like(sc.Psi), linewidth=linewidth, linestyle='--', label='Desired')
    psi_ax.set_xlabel(time_label)
    psi_ax.set_ylabel(r'$\Psi$')
    plt.tight_layout()

    # angular velocity error e_\Omega
    ew_fig, ew_axarr = plt.subplots(3, 1, sharex=True, figsize=scale_figsize(wscale, hscale))
    ew_axarr[0].plot(sc.time, sc.err_vel[:, 0], linewidth=linewidth, linestyle='-', label='Actual')
    ew_axarr[0].plot(sc.time, np.zeros_like(sc.err_vel[:, 0]),
                     linewidth=linewidth, linestyle='--', label='Desired')
    ew_axarr[0].set_ylabel(r'$e_{\Omega_1}$')
    ew_axarr[1].plot(sc.time, sc.err_vel[:, 1], linewidth=linewidth, linestyle='-', label='Actual')
    ew_axarr[1].plot(sc.time, np.zeros_like(sc.err_vel[:, 1]),
                     linewidth=linewidth, linestyle='--', label='Desired')
    ew_axarr[1].set_ylabel(r'$e_{\Omega_2}$')
    ew_axarr[2].plot(sc.time, sc.err_vel[:, 2], linewidth=linewidth, linestyle='-', label='Actual')
    ew_axarr[2].plot(sc.time, np.zeros_like(sc.err_vel[:, 2]),
                     linewidth=linewidth, linestyle='--', label='Desired')
    ew_axarr[2].set_ylabel(r'$e_{\Omega_3}$')
    ew_axarr[2].set_xlabel(time_label)
    plt.tight_layout()

    # plot the control input
    u_fig, u_axarr = plt.subplots(3, 1, sharex=True, figsize=scale_figsize(wscale, hscale))
    u_axarr[0].plot(sc.time, sc.u_m[:, 0], linewidth=linewidth)
    u_axarr[0].set_ylabel(r'$u_1$')
    u_axarr[1].plot(sc.time, sc.u_m[:, 1], linewidth=linewidth)
    u_axarr[1].set_ylabel(r'$u_2$')
    u_axarr[2].plot(sc.time, sc.u_m[:, 2], linewidth=linewidth)
    u_axarr[2].set_ylabel(r'$u_3$')
    u_axarr[2].set_xlabel(time_label)
    plt.tight_layout()

    # angular velocities
    w_fig, w_axarr = plt.subplots(3, 1, sharex=True, figsize=scale_figsize(wscale, hscale))
    w_axarr[0].plot(sc.time, sc.state[:, 9],label=r'Actual', linewidth=linewidth)
    w_axarr[0].plot(sc.time, sc.ang_vel_des[:, 0], label=r'Desired', linewidth=linewidth)
    w_axarr[0].set_ylabel(r'$\Omega_1$')
    w_axarr[1].plot(sc.time, sc.state[:, 10],label=r'Actual', linewidth=linewidth)
    w_axarr[1].plot(sc.time, sc.ang_vel_des[:, 1], label=r'Desired', linewidth=linewidth)
    w_axarr[1].set_ylabel(r'$\Omega_2$')
    w_axarr[2].plot(sc.time, sc.state[:, 11],label=r'Actual', linewidth=linewidth)
    w_axarr[2].plot(sc.time, sc.ang_vel_des[:, 2], label=r'Desired', linewidth=linewidth)
    w_axarr[2].set_ylabel(r'$\Omega_3$')
    plt.tight_layout()

    # disturbance estimates
    delta_actual = np.zeros_like(sc.state[:,12:15])
    for ii,t in enumerate(sc.time):
        delta_actual[ii, :] = sc.delta(t)

    dist_fig, dist_axarr = plt.subplots(3, 1, figsize=scale_figsize(wscale, hscale), sharex=True)
    dist_axarr[0].plot(sc.time, sc.state[:,12], linewidth=linewidth, linestyle='-', label='Estimate')
    dist_axarr[0].plot(sc.time, delta_actual[:,0], linewidth=linewidth,
                       linestyle='--', label='Actual')
    dist_axarr[0].set_ylabel(r'$\bar \Delta_1$')
    dist_axarr[1].plot(sc.time, sc.state[:,13], linewidth=linewidth, linestyle='-', label='Estimate')
    dist_axarr[1].plot(sc.time, delta_actual[:,1], linewidth=linewidth,
                       linestyle='--', label='Actual')
    dist_axarr[1].set_ylabel(r'$\bar \Delta_2$')
    dist_axarr[2].plot(sc.time, sc.state[:,14], linewidth=linewidth, linestyle='-', label='Estimate')
    dist_axarr[2].plot(sc.time, delta_actual[:, 2], linewidth=linewidth,
                       linestyle='--', label='Actual')
    dist_axarr[2].set_ylabel(r'$\bar \Delta_3$')
    dist_axarr[2].set_xlabel(time_label)
    plt.tight_layout()

    # angle to each constraint
    ang_con_fig, ang_con_axarr = plt.subplots(1, 1, figsize=scale_figsize(wscale, hscale))
    ang_con_axarr.plot(sc.time, sc.ang_con, linewidth=linewidth)
    ang_con_axarr.set_xlabel(time_label)
    ang_con_axarr.set_ylabel(r'$\arccos (r^T R^T v_i)$')
    plt.tight_layout()
    # save the figures as pgf if the flag is set
    if pgf_save:
        fig_handles = (er_fig, psi_fig, ew_fig, u_fig, w_fig, dist_fig, ang_con_fig)
        fig_fnames = ('eR', 'Psi', 'eW', 'u', 'w', 'dist', 'ang_con')

        for fig, fname in zip(fig_handles, fig_fnames):
            plt.figure(fig.number)
            plt.savefig(fname + '_' + fname_suffix + '.pgf')
            plt.savefig(fname + '_' + fname_suffix + '.eps', dpi=1200)

    plt.show()

    return 0
