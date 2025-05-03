import numpy as np 
import quadvar as qv
from array import *

# control input
qv.u1 = qv.b*((qv.w1**2)+(qv.w2**2)+(qv.w3**2)+(qv.w4**2))
qv.u2 = qv.b*qv.l*((qv.w3**2)-(qv.w1**2))
qv.u3 = qv.b*qv.l*((qv.w4**2)-(qv.w2**2))
qv.u4 = qv.d*((qv.w2**2)+(qv.w4**2)-(qv.w1**2)-(qv.w3**2))

# quadcopter equation
qv.xdd = (-qv.u1/qv.m) * (np.sin(qv.phi) * np.sin(qv.psi) + np.cos(qv.phi) * np.cos(qv.psi) * np.sin(qv.theta))
qv.ydd = (-qv.u1/qv.m) * (np.cos(qv.phi) * np.sin(qv.psi) * np.sin(qv.theta) - np.cos(qv.psi) * np.sin(qv.phi))
qv.zdd = qv.g - ((qv.u1/qv.m) * (np.cos(qv.phi)*np.cos(qv.theta)))

qv.phidd   = (((qv.iy-qv.iz)/qv.ix) * qv.thetad * qv.psid) + (qv.u2/qv.ix)
qv.thetadd = (((qv.iz-qv.ix)/qv.iy) * qv.phid * qv.psid) + (qv.u3/qv.iy)
qv.psidd   = (((qv.ix-qv.iy)/qv.iz) * qv.phid * qv.thetad) + (qv.u4/qv.iz)

# update XYZ velocity & position
qv.xd = (qv.xdd*qv.ts) + qv.xd
qv.yd = (qv.ydd*qv.ts) + qv.yd
qv.zd = (qv.zdd*qv.ts) + qv.zd

qv.x = (qv.xd*qv.ts) + qv.x
qv.y = (qv.yd*qv.ts) + qv.y
qv.z = (qv.zd*qv.ts) + qv.z

# update phi theta psi angular velocity & position
qv.phid   = (qv.phidd * qv.ts) + qv.phid
qv.thetad = (qv.thetadd * qv.ts) + qv.thetad
qv.psid   = (qv.psidd * qv.ts) + qv.psid

qv.phi   = (qv.phid * qv.ts) + qv.phi
qv.theta = (qv.thetad * qv.ts) + qv.theta
qv.psi   = (qv.psid * qv.ts) + qv.psi

# save data to vector for plotting
qv.x_plot[qv.counter] = qv.x
qv.y_plot[qv.counter] = qv.y
qv.z_plot[qv.counter] = qv.z

qv.phi_plot[qv.counter]   = qv.phi
qv.theta_plot[qv.counter] = qv.theta
qv.psi_plot[qv.counter]   = qv.psi

# update counter
qv.counter = qv.counter + 1