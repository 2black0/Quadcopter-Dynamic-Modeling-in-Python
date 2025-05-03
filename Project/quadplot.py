import quadvar as qv
import numpy as np 
import matplotlib.pyplot as plt

# plot XYZ Positon
plt.figure(1)

plt.subplot(1,3,1)
plt.plot(qv.t_plot,qv.x_plot)
plt.title('X Position')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.grid()

plt.subplot(1,3,2)
plt.plot(qv.t_plot,qv.y_plot)
plt.title('Y Position')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.grid()

plt.subplot(1,3,3)
plt.plot(qv.t_plot,qv.z_plot)
plt.title('Z Position')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.grid()

# plot Phi Theta Psi Angular Positon
plt.figure(2)

plt.subplot(1,3,1)
plt.plot(qv.t_plot,qv.phi_plot)
plt.title('Roll Angle')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.grid()

plt.subplot(1,3,2)
plt.plot(qv.t_plot,qv.theta_plot)
plt.title('Pitch Angle')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.grid()

plt.subplot(1,3,3)
plt.plot(qv.t_plot,qv.psi_plot)
plt.title('Yaw Angle')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.grid()

plt.show()