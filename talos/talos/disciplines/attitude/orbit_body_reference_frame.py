import csdl_alpha as csdl
import numpy as np

# Attitude.py defines an attitude optimization problem for a spacecraft in orbit, using reaction wheels for control, includes effects of gravity gradient torque, integrated in time, and includes constraints on reaction wheel torque and speed, with design variables for the attitude trajectory and initial reaction wheel speeds, and uses CSDL for automatic differentiation and optimization
def orbit_body_reference_frame_change(RTN_from_ECI, B_from_ECI, num_times, step_size):
        ECI_from_RTN = csdl.reorder_axes(RTN_from_ECI, 'ijk->jik')
        B_from_RTN = csdl.einsum(B_from_ECI, ECI_from_RTN, action='ijl,jkl->ikl')
        # Rate of change of Reference frame transformation
        B_from_ECI_dot = csdl.Variable(value = np.zeros((3, 3, num_times)))
        # Next - current time step, divided by step_size gives rate of change
        B_from_ECI_dot = B_from_ECI_dot.set(csdl.slice[:, :, 1:], (B_from_ECI[:, :, 1:] - B_from_ECI[:, :, :-1]) / step_size
        )
        return B_from_RTN, B_from_ECI_dot
