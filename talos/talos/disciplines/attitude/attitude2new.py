import csdl_alpha as csdl
import numpy as np
import matplotlib.pyplot as plt

def attitude_dynamics(omega, B, K, Omega):
    # Euler's equations - gravity gradient torque
    omega_dot_x = K[0] * omega[1] * omega[2] + (-3 * K[0] * B[1, 0] * B[2, 0] * Omega**2)
    omega_dot_y = K[1] * omega[2] * omega[0] + (-3 * K[1] * B[2, 0] * B[0, 0] * Omega**2)
    omega_dot_z = K[2] * omega[0] * omega[1] + (-3 * K[2] * B[0, 0] * B[1, 0] * Omega**2)

    # Build omega_dot as csdl Variable
    omega_dot = csdl.Variable(value=np.zeros(3))
    omega_dot = omega_dot.set(csdl.slice[0], omega_dot_x)
    omega_dot = omega_dot.set(csdl.slice[1], omega_dot_y)
    omega_dot = omega_dot.set(csdl.slice[2], omega_dot_z)

    # Build B_dot as csdl Variable
    B_dot = csdl.Variable(value=np.zeros((3, 3)))
    B_dot = B_dot.set(csdl.slice[0, 2], B[1, 2] * omega[2] - B[2, 2] * omega[1])
    B_dot = B_dot.set(csdl.slice[1, 2], B[2, 2] * omega[0] - B[0, 2] * omega[2])
    B_dot = B_dot.set(csdl.slice[2, 2], B[0, 2] * omega[1] - B[1, 2] * omega[0])
    B_dot = B_dot.set(csdl.slice[0, 0], B[1, 0] * omega[2] - B[2, 0] * omega[1] + Omega * (B[2, 0] * B[1, 2] - B[1, 0] * B[2, 2]))
    B_dot = B_dot.set(csdl.slice[1, 0], B[2, 0] * omega[0] - B[0, 0] * omega[2] + Omega * (B[0, 0] * B[2, 2] - B[2, 0] * B[0, 2]))
    B_dot = B_dot.set(csdl.slice[2, 0], B[0, 0] * omega[1] - B[1, 0] * omega[0] + Omega * (B[1, 0] * B[0, 2] - B[0, 0] * B[1, 2]))

    return omega_dot, B_dot
def runge_kutta_4(f, t0, omega0, B0, h, n, K, Omega):
    omega = omega0
    B = B0
    t = t0

    # preallocate history arrays instead of appending to lists
    omega_history = csdl.Variable(value=np.zeros((n + 1, 3)))
    B_history = csdl.Variable(value=np.zeros((n + 1, 3, 3)))

    # store initial conditions
    omega_history = omega_history.set(csdl.slice[0, :], omega0)
    B_history = B_history.set(csdl.slice[0, :, :], B0)

    for i in csdl.frange(n):
        k1_omega, k1_B = f(omega, B, K, Omega)
        k2_omega, k2_B = f(omega + 0.5*h*k1_omega, B + 0.5*h*k1_B, K, Omega)
        k3_omega, k3_B = f(omega + 0.5*h*k2_omega, B + 0.5*h*k2_B, K, Omega)
        k4_omega, k4_B = f(omega + h*k3_omega, B + h*k3_B, K, Omega)

        omega = omega + (h/6) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
        B = B + (h/6) * (k1_B + 2*k2_B + 2*k3_B + k4_B)

        omega_history = omega_history.set(csdl.slice[i+1, :], omega)
        B_history = B_history.set(csdl.slice[i+1, :, :], B)

    return omega_history, B_history
if __name__ == "__main__":
    h = 0.1
    num_orbits = 10
    num_times = int(2 * np.pi * num_orbits / h)
   
    K1 = -0.5
    K2 = 0.9
    K3 = -(K1 + K2) / (1 + K1 * K2)
    K = np.array([K1, K2, K3])
    Omega = 1.0

    # wrap everything in recorder
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    omega0 = csdl.Variable(value=np.array([0.1*Omega, 0.1*Omega, 1.1*Omega]), name='omega0')
    C0 = np.array([
        [0.9924,  0.0,  -0.0789],
        [-0.0868, 0.0,   0.0944],
        [0.0872,  0.0,   0.9924],
    ])
    c1 = 1.0 - C0[:, 0]**2 - C0[:, 2]**2
    C0[:, 1] = c1
    B0 = csdl.Variable(value=C0, name='B0')

    omega_hist, B_hist = runge_kutta_4(attitude_dynamics, 0.0, omega0, B0, h, num_times, K, Omega)

    recorder.stop()
    t_hist = np.arange(num_times + 1) * h / (2 * np.pi)
    nutation = np.arccos(np.clip(B_hist.value[:, 2, 2], -1, 1)) * 180 / np.pi
    plt.plot(t_hist, nutation)
    plt.title('Nutation angle of spacecraft relative to orbit frame')
    plt.xlabel('Number of orbits')
    plt.ylabel('Nutation angle (degrees)')
    plt.show()

    # plot angular velocity
    plt.plot(t_hist, omega_hist.value[:, 0] * 180 / np.pi * Omega, label='$\\omega_x$')
    plt.plot(t_hist, omega_hist.value[:, 1] * 180 / np.pi * Omega, label='$\\omega_y$')
    plt.plot(t_hist, omega_hist.value[:, 2] * 180 / np.pi * Omega, label='$\\omega_z$')
    plt.title('Angular velocity of spacecraft relative to spacecraft body')
    plt.xlabel('Number of orbits')
    plt.ylabel('Angular velocity (degrees/sec)')
    plt.legend()
    plt.show()