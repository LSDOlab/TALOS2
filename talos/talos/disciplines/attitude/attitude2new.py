import csdl_alpha as csdl
import numpy as np
import matplotlib.pyplot as plt

"""
    Computes the time derivatives of angular velocity (omega_dot) and the
    direction cosine matrix (B_dot) using Euler's equations of motion
    and kinematic equations for a spacecraft in a circular orbit.
    
    Parameters:
        omega : array [wx, wy, wz] - angular velocity of spacecraft body frame
        B     : 3x3 direction cosine matrix (DCM) relating body to orbit frame
        K     : array [K1, K2, K3] - inertia-derived constants for Euler's equations
        Omega : float - orbital rate (rad/s)
    """
def attitude_dynamics(omega, B, K, Omega):
    # Euler's Equations (torque-free rotational dynamics)
    omega_dot_x = K[0] * omega[1] * omega[2]
    omega_dot_y = K[1] * omega[2] * omega[0]
    omega_dot_z = K[2] * omega[0] * omega[1]
  
    # Adds the effect of gravity gradient torque on each axis
    omega_dot_x += -3 * K[0] * (B[1, 0] * B[2, 0] * Omega**2)
    omega_dot_y += -3 * K[1] * (B[2, 0] * B[0, 0] * Omega**2)
    omega_dot_z += -3 * K[2] * (B[0, 0] * B[1, 0] * Omega**2)

    # Column 2 of B_dot: rate of change of the orbit-normal (z-axis) column of DCM
    # This is the standard kinematic equation: dB/dt = omega_cross * B
    B_dot = np.zeros((3, 3))
    B_dot[0, 2] = B[1, 2] * omega[2] - B[2, 2] * omega[1]
    B_dot[1, 2] = B[2, 2] * omega[0] - B[0, 2] * omega[2]
    B_dot[2, 2] = B[0, 2] * omega[1] - B[1, 2] * omega[0]
    # Column 0 of B_dot: rate of change of the orbit radial (x-axis) column of DCM
    # Includes extra terms involving Omega because the orbit frame itself is rotating
    # The Omega terms account for the fact that the reference orbit frame rotates
    # around Earth at the orbital rate
    B_dot[0, 0] = B[1, 0] * omega[2] - B[2, 0] * omega[1] + Omega * (B[2, 0] * B[1, 2] - B[1, 0] * B[2, 2])
    B_dot[1, 0] = B[2, 0] * omega[0] - B[0, 0] * omega[2] + Omega * (B[0, 0] * B[2, 2] - B[2, 0] * B[0, 2])
    B_dot[2, 0] = B[0, 0] * omega[1] - B[1, 0] * omega[0] + Omega * (B[1, 0] * B[0, 2] - B[0, 0] * B[1, 2])
 #^ rewrite to use CSDL syntax
    return np.array([omega_dot_x, omega_dot_y, omega_dot_z]), B_dot
    """
      Parameters:
        f      : the dynamics function (attitude_dynamics)
        t0     : initial time
        omega0 : initial angular velocity vector
        B0     : initial direction cosine matrix
        h      : time step size (seconds)
        n      : number of time steps
        K      : inertia constants
        Omega  : orbital rate
    """

def runge_kutta_4(f, t0, omega0, B0, h, n, K, Omega):
    omega = omega0
   #initialize as CSDL array
    B = B0
    t = t0

    omega_history = [omega0]
    B_history = [B0]

    for _ in range(n):
        k1_omega, k1_B = f(omega, B, K, Omega)
        k2_omega, k2_B = f(omega + 0.5*h*k1_omega, B + 0.5*h*k1_B, K, Omega)
        k3_omega, k3_B = f(omega + 0.5*h*k2_omega, B + 0.5*h*k2_B, K, Omega)
        k4_omega, k4_B = f(omega + h*k3_omega, B + h*k3_B, K, Omega)

        omega = omega + (h/6) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
        B = B + (h/6) * (k1_B + 2*k2_B + 2*k3_B + k4_B)
# histories need to be csdl, instead of making a list preallocate arrays of correct size
# RK4 instead of appending to lists, we can directly assign to preallocated arrays
# syntax for handling B and RK4 function
        omega_history.append(omega)
        B_history.append(B)

    return np.array(omega_history), np.array(B_history)

# run code will have to change, doesnt run code, just generates
if __name__ == "__main__":
    h = 0.1 # time step size in normalized orbital time units
    num_orbits = 10 # total number of orbits to simulate
    num_times = int(2 * np.pi * num_orbits / h) # total number of time steps (one orbit is 2*pi in normalized time)
   
    # Inertia Constants (K values)
    # K1 and K2 come from the principal moments of inertia: K = (Ij - Ik) / Ii
    # K3 is derived to satisfy the constraint K1*K2*K3 relationship for stability
    K1 = -0.5
    K2 = 0.9
    K3 = -(K1 + K2) / (1 + K1 * K2)
    K = np.array([K1, K2, K3])
    # Orbital rate
    Omega = 1.0
    # Initial angular velocity (rad/s) - starting with a small initial spin around the z-axis and small perturbations in x and y
    omega0 = np.array([0.1*Omega, 0.1*Omega, 1.1*Omega])
    # Initial Direction Cosine Matrix (DCM)
    # C0 maps from the body frame to the orbit frame
    C0 = np.array([
        [0.9924,  0.0,  -0.0789],
        [-0.0868, 0.0,   0.0944],
        [0.0872,  0.0,   0.9924],
    ])
    c1 = 1.0 - C0[:, 0]**2 - C0[:, 2]**2
    C0[:, 1] = c1

    omega_hist, B_hist = runge_kutta_4(attitude_dynamics, 0.0, omega0, C0, h, num_times, K, Omega)

    t_hist = np.arange(num_times + 1) * h / (2 * np.pi)

    # plot nutation angle
    nutation = np.arccos(np.clip(B_hist[:, 2, 2], -1, 1)) * 180 / np.pi
    plt.plot(t_hist, nutation)
    plt.title('Nutation angle of spacecraft relative to orbit frame')
    plt.xlabel('Number of orbits')
    plt.ylabel('Nutation angle (degrees)')
    plt.show()

    # plot angular velocity
    plt.plot(t_hist, omega_hist[:, 0] * 180 / np.pi * Omega, label='$\\omega_x$')
    plt.plot(t_hist, omega_hist[:, 1] * 180 / np.pi * Omega, label='$\\omega_y$')
    plt.plot(t_hist, omega_hist[:, 2] * 180 / np.pi * Omega, label='$\\omega_z$')
    plt.title('Angular velocity of spacecraft relative to spacecraft body')
    plt.xlabel('Number of orbits')
    plt.ylabel('Angular velocity (degrees/sec)')
    plt.legend()
    plt.show()