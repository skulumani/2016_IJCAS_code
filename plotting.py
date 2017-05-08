"""Plot the outputs of the simulation

"""
import numpy as np

import matplotlib 
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import colors, animation, cm

figsize = (8,6)
time_label = r'$t$'
# parse out the simulation results and recompute the error functions for 

def plot_outputs(sc):
    """Given a simulated rigid body instantiation this will plot all the outputs
    
    Parameters:
    ----------
    time: (n,) numpy array
    sc: rigid-body object

    """
    # plot eR components
    er_fig, er_axarr = plt.subplots(3,1, sharex=True, figsize=figsize)
    er_axarr[0].plot(sc.time, sc.err_att[:,0])
    er_axarr[0].set_ylabel(r'$e_{R_1}$')
    er_axarr[1].plot(sc.time, sc.err_att[:,1])
    er_axarr[1].set_ylabel(r'$e_{R_2}$')
    er_axarr[2].plot(sc.time, sc.err_att[:,2])
    er_axarr[2].set_ylabel(r'$e_{R_3}$')
    er_axarr[2].set_xlabel(time_label)

    # plot configuration error function \Psi
    psi_fig, psi_ax = plt.subplots(1, 1, figsize=figsize)
    psi_ax.plot(sc.time, sc.Psi)
    psi_ax.set_xlabel(time_label)
    psi_ax.set_ylabel(r'$\Psi$')

    # angular velocity error e_\Omega
    ew_fig, ew_axarr = plt.subplots(3, 1, sharex=True, figsize=figsize)
    ew_axarr[0].plot(sc.time, sc.err_vel[:, 0])
    ew_axarr[0].set_ylabel(r'$e_{\Omega_1}$')
    ew_axarr[1].plot(sc.time, sc.err_vel[:, 1])
    ew_axarr[1].set_ylabel(r'$e_{\Omega_2}$')
    ew_axarr[2].plot(sc.time, sc.err_vel[:, 2])
    ew_axarr[2].set_ylabel(r'$e_{\Omega_3}$')
    ew_axarr[2].set_xlabel(time_label)

    # plot the control input
    u_fig, u_axarr = plt.subplots(3, 1, sharex=True, figsize=figsize)
    u_axarr[0].plot(sc.time, sc.u_m[:, 0])
    u_axarr[0].set_ylabel(r'$u_1$')
    u_axarr[1].plot(sc.time, sc.u_m[:, 1])
    u_axarr[1].set_ylabel(r'$u_2$')
    u_axarr[2].plot(sc.time, sc.u_m[:, 2])
    u_axarr[2].set_ylabel(r'$u_3$')
    u_axarr[2].set_xlabel(time_label)

    # angular velocities
    w_fig, w_axarr = plt.subplots(3, 1, sharex=True, figsize=figsize)
    w_axarr[0].plot(sc.time, sc.state[:, 9],label=r'Actual')
    w_axarr[0].plot(sc.time, sc.ang_vel_des[:, 0], label=r'Desired')
    w_axarr[0].set_ylabel(r'$\Omega_1$')
    w_axarr[1].plot(sc.time, sc.state[:, 10],label=r'Actual')
    w_axarr[1].plot(sc.time, sc.ang_vel_des[:, 1], label=r'Desired')
    w_axarr[1].set_ylabel(r'$\Omega_2$')
    w_axarr[2].plot(sc.time, sc.state[:, 11],label=r'Actual')
    w_axarr[2].plot(sc.time, sc.ang_vel_des[:, 2], label=r'Desired')
    w_axarr[2].set_ylabel(r'$\Omega_3$')

    # disturbance estimates
    dist_fig, dist_axarr = plt.subplots(1, 1, figsize=figsize)
    dist_axarr.plot(sc.time, sc.state[:,12:15])
    dist_axarr.set_xlabel(time_label)
    dist_axarr.set_ylabel(r'$\bar \Delta$')

    # angle to each constraint
    ang_con_fig, ang_con_axarr = plt.subplots(1, 1, figsize=figsize)
    ang_con_axarr.plot(sc.time, sc.ang_con)
    ang_con_axarr.set_xlabel(time_label)
    ang_con_axarr.set_ylabel(r'$\arccos (r^T R^T v_i)$')

    plt.show()

    return 0
