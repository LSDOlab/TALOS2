import csdl_alpha as csdl
import numpy as np
import matplotlib.pyplot as plt

# Attitude2new.py defines attitude dynamics of spacecraft in orbit, includes effects of gravity gradient torque, integrates in time, plots results for nutation angles and angular velocity
def attitude_dynamics(omega, B, K, Omega):
    # Euler's equations - gravity gradient torque
    # First term (K[0] * omega[1] * omega[2]) is the gyroscopic term, second term is the gravity gradient torque
    # K is related to the moments of inertia, and B is the direction cosine matrix from the orbit frame to the body frame
    omega_dot_x = K[0] * omega[1] * omega[2] + (-3 * K[0] * B[1, 0] * B[2, 0] * Omega**2)
    omega_dot_y = K[1] * omega[2] * omega[0] + (-3 * K[1] * B[2, 0] * B[0, 0] * Omega**2)
    omega_dot_z = K[2] * omega[0] * omega[1] + (-3 * K[2] * B[0, 0] * B[1, 0] * Omega**2)

    # Build omega_dot as csdl Variable
    # Created a csdl Variable initialized to zeros, then fill it slot by slot, building arrays while keeping operations differentiable
    # CSDL needs to track each assignment
    omega_dot = csdl.Variable(value=np.zeros(3))
    omega_dot = omega_dot.set(csdl.slice[0], omega_dot_x)
    omega_dot = omega_dot.set(csdl.slice[1], omega_dot_y)
    omega_dot = omega_dot.set(csdl.slice[2], omega_dot_z)

    # Build B_dot as csdl Variable
    # DCM Rate of Change, Direction Cosine Matrix, 3x3 rotation matrix describing how spacecraft body frame relates to orbital frame
    # Three lines update column 2 of B, which is the direction of the orbit normal in the body frame, and the other six lines update
    # columns 0 and 1, which are the directions of the velocity and position vectors in the body frame, the extra Omega *() 
    # term comes from the orbital frame itself rotating at rate Omega around earth, so the DCM rate of change has contributions from
    # both the spacecraft's own rotation and the rotation of the orbital frame
    B_dot = csdl.Variable(value=np.zeros((3, 3)))
    B_dot = B_dot.set(csdl.slice[0, 2], B[1, 2] * omega[2] - B[2, 2] * omega[1])
    B_dot = B_dot.set(csdl.slice[1, 2], B[2, 2] * omega[0] - B[0, 2] * omega[2])
    B_dot = B_dot.set(csdl.slice[2, 2], B[0, 2] * omega[1] - B[1, 2] * omega[0])
    B_dot = B_dot.set(csdl.slice[0, 0], B[1, 0] * omega[2] - B[2, 0] * omega[1] + Omega * (B[2, 0] * B[1, 2] - B[1, 0] * B[2, 2]))
    B_dot = B_dot.set(csdl.slice[1, 0], B[2, 0] * omega[0] - B[0, 0] * omega[2] + Omega * (B[0, 0] * B[2, 2] - B[2, 0] * B[0, 2]))
    B_dot = B_dot.set(csdl.slice[2, 0], B[0, 0] * omega[1] - B[1, 0] * omega[0] + Omega * (B[1, 0] * B[0, 2] - B[0, 0] * B[1, 2]))
    # Returns both derivatives so the integrator can use them
    return omega_dot, B_dot
    # Integrates the dynamics forward in time, given a starting point, stepping 
    # forward to build the full trajectory
def runge_kutta_4(f, t0, omega0, B0, h, n, K, Omega):
   # Initialization of the state variables for the integrator, and preallocation of history arrays to store the trajectory.
    omega = omega0
    B = B0
    t = t0
    omega_history = csdl.Variable(value=np.zeros((n + 1, 3)))
    B_history = csdl.Variable(value=np.zeros((n + 1, 3, 3)))
    omega_history = omega_history.set(csdl.slice[0, :], omega0)
    B_history = B_history.set(csdl.slice[0, :, :], B0)
    # csdl.frange is graph aware version of range, allowing csdl to record what happens in each iteration of the loop, and
    # allowing us to build the history arrays in a way that csdl can track and differentiate through
    for i in csdl.frange(n):
    # k1 derivative at the current state, k2 at the midpoint using k1, k3 at the midpoint using k2, and k4 at the end using k3, then combine them to get the next state
        k1_omega, k1_B = f(omega, B, K, Omega)
        k2_omega, k2_B = f(omega + 0.5*h*k1_omega, B + 0.5*h*k1_B, K, Omega)
        k3_omega, k3_B = f(omega + 0.5*h*k2_omega, B + 0.5*h*k2_B, K, Omega)
        k4_omega, k4_B = f(omega + h*k3_omega, B + h*k3_B, K, Omega)
    # Update the state using the weighted average of the derivatives
        omega = omega + (h/6) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
        B = B + (h/6) * (k1_B + 2*k2_B + 2*k3_B + k4_B)
    # Saved updated state into history arrays at index i+1, since index 0 is the initial condition
        omega_history = omega_history.set(csdl.slice[i+1, :], omega)
        B_history = B_history.set(csdl.slice[i+1, :, :], B)

    return omega_history, B_history
if __name__ == "__main__":
    h = 0.1
    num_orbits = 10
    num_times = int(2 * np.pi * num_orbits / h)
   # Inertia ratios, encoding the shape of the spacecraft
    K1 = -0.5
    K2 = 0.9
    K3 = -(K1 + K2) / (1 + K1 * K2)
    K = np.array([K1, K2, K3])
    # Normalizes orbital rate to 1, all time measured in units of orbital periods
    Omega = 1.0
    # Wrap everything in recorder
    # Recorder is csdl's mechanism for capturing the computational graph
    # inLine=True means values are computed immediately as operations are defined
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    # Initial angular velocity and DCM, with small initial nutation, so the spacecraft isn't perfectly aligned with the orbit frame
    omega0 = csdl.Variable(value=np.array([0.1*Omega, 0.1*Omega, 1.1*Omega]), name='omega0')
    # C0 is the initial DCM, column 0 and 2 set manually, column 1 computed so each row has unit magnitude
    # Wrapped as csdl Variable so influence on outputs can be differentiated
    C0 = np.array([
        [0.9924,  0.0,  -0.0789],
        [-0.0868, 0.0,   0.0944],
        [0.0872,  0.0,   0.9924],
    ])
    c1 = 1.0 - C0[:, 0]**2 - C0[:, 2]**2
    C0[:, 1] = c1
    B0 = csdl.Variable(value=C0, name='B0')
    # Runs full sim inside of recorder, building complete computational graph
    omega_hist, B_hist = runge_kutta_4(attitude_dynamics, 0.0, omega0, B0, h, num_times, K, Omega)
    recorder.stop()
    # Converts step indices to orbit numbers for x-axis
    t_hist = np.arange(num_times + 1) * h / (2 * np.pi)
    # Plots the nutation angle, which is the angle between the spacecraft's z-axis and the orbit normal, 
    # computed from the DCM as arccos of the (2,2) element, which is the cosine of that angle
    nutation = np.arccos(np.clip(B_hist.value[:, 2, 2], -1, 1)) * 180 / np.pi
    # Plot nutation angle over time
    plt.plot(t_hist, nutation)
    plt.title('Nutation angle of spacecraft relative to orbit frame')
    plt.xlabel('Number of orbits')
    plt.ylabel('Nutation angle (degrees)')
    plt.show()

    # Plot angular velocity
    plt.plot(t_hist, omega_hist.value[:, 0] * 180 / np.pi * Omega, label='$\\omega_x$')
    plt.plot(t_hist, omega_hist.value[:, 1] * 180 / np.pi * Omega, label='$\\omega_y$')
    plt.plot(t_hist, omega_hist.value[:, 2] * 180 / np.pi * Omega, label='$\\omega_z$')
    plt.title('Angular velocity of spacecraft relative to spacecraft body')
    plt.xlabel('Number of orbits')
    plt.ylabel('Angular velocity (degrees/sec)')
    plt.legend()
    plt.show()