import numpy as np 
from array import *

# for simulation
sim_time = 30
ts       = .01
t_plot   = np.arange(0,sim_time-ts,ts)
counter  = 0

# global constants for quadrotor
g   = 9.81
b   = 3.25e-5
d   = 7.5e-7
l   = 0.25
m   = 0.65
ix  = 7.5e-3
iy  = 7.5e-3
iz  = 1.3e-2

# angular speed of-i motor
w1    = np.sqrt(m*g/(4*b))
w2    = np.sqrt(m*g/(4*b))
w3    = np.sqrt(m*g/(4*b))
w4    = np.sqrt(m*g/(4*b))
wmax  = 1000

# initial condition
x       = 0
y       = 0
z       = 0
xd      = 0
yd      = 0
zd      = 0
xdd     = 0
ydd     = 0
zdd     = 0

phi     = 0
theta   = 0
psi     = 0
phid    = 0
thetad  = 0
psid    = 0
phidd   = 0
thetadd = 0
psidd   = 0

# control input
u1 = 0
u2 = 0
u3 = 0
u4 = 0

# plot variable
x_plot = np.zeros(len(t_plot-1))
y_plot = np.zeros(len(t_plot-1))
z_plot = np.zeros(len(t_plot-1))

phi_plot   = np.zeros(len(t_plot-1))
theta_plot = np.zeros(len(t_plot-1))
psi_plot   = np.zeros(len(t_plot-1))