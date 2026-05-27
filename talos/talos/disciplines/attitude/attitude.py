import csdl_alpha as csdl
import numpy as np
from talos.utils.bspline_comp import BsplineComp, get_bspline_mtx
from talos.disciplines.reference_frames.body123 import body123_reference_frame_change
from talos.disciplines.attitude.orbit_body_reference_frame import orbit_body_reference_frame_change


def body_rates(B_from_ECI, B_from_ECI_dot, osculating_orbit_angular_speed,
               sc_mmoi, step_size, num_times, gravity_gradient, B_from_RTN=None):
    # Angular velocity via skew-symmetric cross operator: wcross = B_dot @ B^T
    wcross = csdl.einsum(
        B_from_ECI_dot,
        csdl.einsum(B_from_ECI, action='ijk->jik'),
        action='ijl,jkl->ikl')

    # Extract angular velocity components from skew-symmetric matrix
    rates = csdl.Variable(value=np.zeros((num_times, 3)))
    rates = rates.set(csdl.slice[:, 0], wcross[2, 1, :])
    rates = rates.set(csdl.slice[:, 1], wcross[0, 2, :])
    rates = rates.set(csdl.slice[:, 2], wcross[1, 0, :])

    # Angular acceleration via finite differences
    accels = csdl.Variable(value=np.zeros((num_times, 3)))
    accels = accels.set(
        csdl.slice[1:, :],
        (rates[1:, :] - rates[:-1, :]) / step_size
    )

    # Jw = angular momentum = J * omega
    Jw = rates * np.einsum('i,j->ij', np.ones(num_times), sc_mmoi)

    # bt1 = J * alpha (inertial torque)
    bt1 = csdl.Variable(value=np.zeros((num_times, 3)))
    bt1 = bt1.set(csdl.slice[:, 0], sc_mmoi[0] * accels[:, 0])
    bt1 = bt1.set(csdl.slice[:, 1], sc_mmoi[1] * accels[:, 1])
    bt1 = bt1.set(csdl.slice[:, 2], sc_mmoi[2] * accels[:, 2])

    # bt2 = omega x (J * omega) (gyroscopic torque)
    bt2 = csdl.cross(rates, Jw, axis=1)

    # B_from_RTN is the rotation matrix that transforms vectors from the RTN to Body frame
    # RTN - Radial points away from Earth, Tangential points in the direction of motion along orbit, Normal points perpendicularly to the orbital plane
    # Gravity gradient torque arises from the non-uniform gravity field of Earth
    # T = -3 * (I2 - I3) * B_from_RTN[1,0] * B_from_RTN[2,0] * n^2 for x-axis
    # where n is the orbital angular velocity, with cyclic permutations for y and z axes
    if gravity_gradient is True:
        bt3 = csdl.Variable(value=np.zeros((3, num_times)))
        bt3 = bt3.set(csdl.slice[0, :],
            -3 * (sc_mmoi[1] - sc_mmoi[2]) * B_from_RTN[1, 0, :] * B_from_RTN[2, 0, :] * osculating_orbit_angular_speed[0, :]**2)
        bt3 = bt3.set(csdl.slice[1, :],
            -3 * (sc_mmoi[2] - sc_mmoi[0]) * B_from_RTN[2, 0, :] * B_from_RTN[0, 0, :] * osculating_orbit_angular_speed[0, :]**2)
        bt3 = bt3.set(csdl.slice[2, :],
            -3 * (sc_mmoi[0] - sc_mmoi[1]) * B_from_RTN[0, 0, :] * B_from_RTN[1, 0, :] * osculating_orbit_angular_speed[0, :]**2)
        body_torque = bt1 + bt2 + csdl.reorder_axes(bt3, 'ij->ji')
    else:
        body_torque = bt1 + bt2

    return rates, body_torque


def reaction_wheel_dynamics(omega, body_rates, body_torque, rw_mmoi):
    # x = J_rw * omega_body (angular momentum of reaction wheels)
    x = csdl.Variable(value=np.zeros(3))
    x = x.set(csdl.slice[0], rw_mmoi[0] * body_rates[0])
    x = x.set(csdl.slice[1], rw_mmoi[1] * body_rates[1])
    x = x.set(csdl.slice[2], rw_mmoi[2] * body_rates[2])
    # dw_dt = omega_body x (J_rw * omega_body) - body_torque
    dw_dt = csdl.cross(body_rates, x, axis=0) - body_torque
    return dw_dt


def runge_kutta_4(f, omega0, body_rates_history, body_torque_history, h, n, rw_mmoi):
    # Preallocate history array and set initial condition
    omega = omega0
    omega_history = csdl.Variable(value=np.zeros((n + 1, 3)))
    omega_history = omega_history.set(csdl.slice[0, :], omega0)

    for i in csdl.frange(n):
        # Look up spacecraft angular velocity and torque at this time step
        B = body_rates_history[i, :]
        torque = body_torque_history[i, :]

        # RK4 derivative estimates
        k1 = f(omega, B, torque, rw_mmoi)
        k2 = f(omega + 0.5*h*k1, B, torque, rw_mmoi)
        k3 = f(omega + 0.5*h*k2, B, torque, rw_mmoi)
        k4 = f(omega + h*k3, B, torque, rw_mmoi)

        # Weighted average update
        omega = omega + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        omega_history = omega_history.set(csdl.slice[i+1, :], omega)

    return omega_history


def attitude(num_times, num_cp, step_size, RTN_from_ECI, osculating_orbit_angular_speed,
             max_rw_torque=0.004,
             sc_mmoi=6 * np.array([2, 1, 3]) * 1e-3,
             rw_mmoi=6 * np.ones(3) * 1e-5,
             gravity_gradient=True):

    if sc_mmoi.shape != (3,):
        raise ValueError('sc_mmoi must have shape (3,); has shape {}'.format(sc_mmoi.shape))
    if rw_mmoi.shape != (3,):
        raise ValueError('rw_mmoi must have shape (3,); has shape {}'.format(rw_mmoi.shape))

    max_rw_speed = 1.0 / max_rw_torque

    # B-spline control points (design variables)
    jac = get_bspline_mtx(num_cp, num_times)
    # setting values to test
    yaw_cp = csdl.Variable(value=np.linspace(0, 0.1, num_cp), name='yaw_cp')
    pitch_cp = csdl.Variable(value=np.linspace(0, 0.05, num_cp), name='pitch_cp')
    roll_cp = csdl.Variable(value=np.linspace(0, 0.05, num_cp), name='roll_cp')
    # yaw_cp = csdl.Variable(value=np.zeros(num_cp), name='yaw_cp')
    # pitch_cp = csdl.Variable(value=np.zeros(num_cp), name='pitch_cp')
    # roll_cp = csdl.Variable(value=np.zeros(num_cp), name='roll_cp')
    # scale these
    yaw_cp.set_as_design_variable(scaler=1e10)
    pitch_cp.set_as_design_variable(scaler=1e10)
    roll_cp.set_as_design_variable(scaler=1e10)

    yaw_inputs = csdl.VariableGroup()
    yaw_inputs.yaw_cp = yaw_cp
    yaw = BsplineComp(num_cp=num_cp, num_pt=num_times, jac=jac,
                      in_name='yaw_cp', out_name='yaw').evaluate(yaw_inputs).yaw

    pitch_inputs = csdl.VariableGroup()
    pitch_inputs.pitch_cp = pitch_cp
    pitch = BsplineComp(num_cp=num_cp, num_pt=num_times, jac=jac,
                        in_name='pitch_cp', out_name='pitch').evaluate(pitch_inputs).pitch

    roll_inputs = csdl.VariableGroup()
    roll_inputs.roll_cp = roll_cp
    roll = BsplineComp(num_cp=num_cp, num_pt=num_times, jac=jac,
                       in_name='roll_cp', out_name='roll').evaluate(roll_inputs).roll
    # set final constraints for attitude to force optimization problem towards final target orientation minimizing torque along the way
    # anytime you add a constraint add a scaler to make it order of magnitude 1
    #yaw[-1].set_as_constraint(equals=0.1, scaler=10.0)
    #roll[-1].set_as_constraint(equals=0.05, scaler=20.0)
    #pitch[-1].set_as_constraint(equals=0.05, scaler=20.0)
    yaw[-1].set_as_constraint(equals=0)
    roll[-1].set_as_constraint(equals=0)
    pitch[-1].set_as_constraint(equals=0)
    # Reference frame transformations
    B_from_ECI = body123_reference_frame_change(yaw, pitch, roll, num_times)
    B_from_RTN, B_from_ECI_dot = orbit_body_reference_frame_change(RTN_from_ECI, B_from_ECI, num_times, step_size)
    rates, body_torque = body_rates(B_from_ECI, B_from_ECI_dot, osculating_orbit_angular_speed,
                                    sc_mmoi, step_size, num_times, gravity_gradient, B_from_RTN)

    # Initial reaction wheel velocity (design variable)
    initial_rw_velocity = csdl.Variable(value=np.zeros(3), name='initial_reaction_wheel_velocity')
   # initial_rw_velocity.set_as_design_variable(lower=-max_rw_speed, upper=max_rw_speed)

    # RK4 integration of reaction wheel dynamics
    rw_velocity_history = runge_kutta_4(
        reaction_wheel_dynamics,
        initial_rw_velocity,
        rates,
        body_torque,
        step_size,
        num_times - 1,
        rw_mmoi
    )

    # Reaction wheel acceleration and torque
    rw_accel_history = csdl.Variable(value=np.zeros((num_times, 3)))
    rw_accel_history = rw_accel_history.set(
        csdl.slice[1:, :],
        (rw_velocity_history[1:, :] - rw_velocity_history[:-1, :]) / step_size
    )
    reaction_wheel_torque = csdl.Variable(value=np.zeros((num_times, 3)))
    reaction_wheel_torque = reaction_wheel_torque.set(csdl.slice[:, 0], rw_mmoi[0] * rw_accel_history[:, 0])
    reaction_wheel_torque = reaction_wheel_torque.set(csdl.slice[:, 1], rw_mmoi[1] * rw_accel_history[:, 1])
    reaction_wheel_torque = reaction_wheel_torque.set(csdl.slice[:, 2], rw_mmoi[2] * rw_accel_history[:, 2])

    # Constraints on reaction wheel torque
    # min and max need to be adjusted
    min_rw_torque = csdl.minimum(reaction_wheel_torque)
    max_rw_torque_val = csdl.maximum(reaction_wheel_torque)
    min_rw_torque.set_as_constraint(lower=-max_rw_torque, scaler=1e150)
    max_rw_torque_val.set_as_constraint(upper=max_rw_torque, scaler=1e150)
    rw_effort = csdl.sum(reaction_wheel_torque ** 2)
    rw_effort.set_as_objective(scaler=1e20)
    return rw_velocity_history, reaction_wheel_torque, yaw, pitch, roll


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.random.seed(0)

    num_times = 301
    num_cp = int((num_times - 1) / 5)
    duration = 95.
    step_size = duration * 60 / (num_times - 1)

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    RTN_from_ECI = csdl.Variable(
        value=np.tile(np.eye(3)[:, :, np.newaxis], (1, 1, num_times)),
        name='RTN_from_ECI'
    )
    osculating_orbit_angular_speed = csdl.Variable(
        value=np.ones((1, num_times)) * 0.001,
        name='osculating_orbit_angular_speed'
    )

    rw_vel, rw_torque, yaw, pitch, roll = attitude(
        num_times=num_times,
        num_cp=num_cp,
        step_size=step_size,
        RTN_from_ECI=RTN_from_ECI,
        osculating_orbit_angular_speed=osculating_orbit_angular_speed,
        gravity_gradient=True,
    )

    recorder.stop()

    # sim = csdl.experimental.JaxSimulator(recorder=recorder)
    # sim.check_totals()
    sim = csdl.experimental.PySimulator(recorder)
    sim.run()

    from modopt import CSDLAlphaProblem, SLSQP
    prob = CSDLAlphaProblem(problem_name='attitude_opt', simulator=sim)
    optimizer = SLSQP(prob, solver_options={'ftol': 1e-9, 'maxiter': 100})


    #print("Initial design variables:", prob.x0)
   # print("Initial objective:", prob.f_s)
    #print("Initial constraints:", prob.c_s)

    optimizer.solve()
    optimizer.print_results()

    print("Reaction wheel velocity history shape:", rw_vel.value.shape)
    print("First RW velocity:", rw_vel.value[0, :])
    print("Last RW velocity:", rw_vel.value[-1, :])

    # Time axis in minutes
    t_hist = np.arange(num_times) * step_size / 60
    labels = ['x', 'y', 'z']

    # Plot reaction wheel velocity
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    for i in range(3):
        ax[i].plot(t_hist, rw_vel.value[:, i])
        ax[i].set_ylabel(f'RW velocity {labels[i]} (rad/s)')
        ax[i].grid(True)
    ax[-1].set_xlabel('Time (minutes)')
    ax[0].set_title('Reaction Wheel Velocity History')
    plt.tight_layout()
    plt.show()

    # Plot reaction wheel torque
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    for i in range(3):
        ax[i].plot(t_hist, rw_torque.value[:, i])
        ax[i].set_ylabel(f'RW torque {labels[i]} (Nm)')
        ax[i].grid(True)
    ax[-1].set_xlabel('Time (minutes)')
    ax[0].set_title('Reaction Wheel Torque History')
    plt.tight_layout()
    plt.show()

    # Plot yaw, pitch, roll
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    for i, (angle, name) in enumerate(zip([yaw, pitch, roll], ['Yaw', 'Pitch', 'Roll'])):
        ax[i].plot(t_hist, np.degrees(angle.value))
        ax[i].set_ylabel(f'{name} (degrees)')
        ax[i].grid(True)
    ax[-1].set_xlabel('Time (minutes)')
    ax[0].set_title('Attitude Angles')
    plt.tight_layout()
    plt.show()