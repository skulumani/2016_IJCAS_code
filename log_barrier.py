import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
from matplotlib import colors, animation, cm
from mpl_toolkits.mplot3d import Axes3D

from kinematics import attitude
# define LaTeX figure properties 

def figsize(scale):
    fig_width_pt = 469.75502                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def scale_figsize(wscale, hscale):
    fig_width_pt = 469.75502
    fig_height_pt = 650.43001
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
        "axes.labelsize": 10,               # LaTeX default is 10pt font.
        "font.size": 10,
        "legend.fontsize": 10,               # Make the legend/label fonts a little smaller
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
            r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
            r"\usepackage{siunitx}",
            ]
        }

matplotlib.rcParams.update(pgf_with_latex)

con_angle = np.cos(10 * np.pi/180)
angle = np.arange(-1, con_angle, 1e-2)
x = np.arange(0, 1, 1e-3)
alpha = 20

sen = np.array([1, 0, 0])
con = np.array([1, 1, 0])
con = con / np.linalg.norm(con)

R_des = np.eye(3,3)
G = np.diag([0.9, 1.0, 1.1])

# cylindrical projection of 2-Sphere
den = 50
lon = np.linspace(-180, 180, den) * np.pi/180
lat = np.linspace(-90, 90, den) * np.pi/180

X, Y = np.meshgrid(lon, lat)

sen_array = np.zeros((3, den**2))
er_attract_array = np.zeros_like(sen_array)
psi_avoid_array = np.zeros_like(X)
psi_attract_array = np.zeros_like(X)
psi_total_array = np.zeros_like(X)

# now loop over all the angles and compute the error function
for ii in range(lon.shape[0]):
    for jj in range(lat.shape[0]):
        # rotate the body vector to the inertial frame
        R_b2i = attitude.rot2(lat[jj]).T.dot(attitude.rot3(lon[ii]))

        sen_inertial = R_b2i.dot(sen)
        sen_array[:, ii*den + 0] = sen_inertial

        psi_attract = 1/2 * np.trace(G.dot(np.eye(3, 3) - R_des.T.dot(R_b2i)))

        if np.dot(sen_inertial, con) < con_angle:
            psi_avoid = -1 / alpha * np.log(- (np.dot(sen_inertial, con) - con_angle) / (1 + con_angle)) + 1
        else:
            psi_avoid = 3

        psi_total = psi_attract * psi_avoid

        er = 1/2 * attitude.vee_map(G.dot(R_des.T).dot(R_b2i) - R_b2i.T.dot(R_des).dot(G))

        er_attract_array[:, ii*den + jj] = er.T
        psi_avoid_array[ii, jj] = psi_avoid
        psi_attract_array[ii, jj] = psi_attract
        psi_total_array[ii, jj] = psi_total


g = 1 + -1/alpha * np.log(- (angle - con_angle) / (1+con_angle))
eRB = np.absolute(1 / alpha * np.sin(np.arccos(angle)) / (angle - con_angle))

# Log barrier function plot
lb_fig, lb_ax = plt.subplots(1, 1, figsize=figsize(1))
lb_ax.plot(np.arccos(angle) * 180/np.pi, np.real(g), label='Barrier Function')
lb_ax.plot(np.arccos(angle) * 180/np.pi, eRB, label=r'$e_{R_B}$')
lb_ax.set_xlabel('Angle to Constraint')
lb_ax.set_ylabel('Error Function')
lb_ax.set_title('Logarithmic Barrier Function')

# Error function surface visualization
fwidth = 1
avoid_fig, avoid_ax = plt.subplots(1, 1, figsize=figsize(fwidth))
avoid_ax = Axes3D(avoid_fig)
avoid_ax.plot_surface(X * 180/np.pi, Y * 180/np.pi, psi_avoid_array)
avoid_ax.set_xlim(-180, 180)
avoid_ax.set_ylim(-90, 90)
avoid_ax.set_zlim(0, 3)
plt.show()


