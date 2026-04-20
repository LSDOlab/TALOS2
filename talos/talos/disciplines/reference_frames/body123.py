import csdl_alpha as csdl
import numpy as np

# Update for new CSDL version, using csdl.Variable instead of self.declare_variable, and using csdl.slice to build up the output variable slot by slot, since CSDL needs to track each assignment to the variable
def body123_reference_frame_change(yaw, pitch, roll, num_times):
    # Computes 123 rotation matrix C, (shape 3, 3, num_times) describing how to express coordinates in the body frame that were originally defined in the ECI (Earth Centered Inertial) frame, given time histories of yaw, pitch, and roll angles (shape (num_times, ))
    sr = csdl.sin(roll)
    cr = csdl.cos(roll)
    sp = csdl.sin(pitch)
    cp = csdl.cos(pitch)
    sy = csdl.sin(yaw)
    cy = csdl.cos(yaw)

    # Create a tracked output variable and fill it slot-by-slot using csdl.slice.
    # New CSDL will broadcast the (num_times,) arrays automatically, so no csdl.expand is needed.
    C = csdl.Variable(shape=(3, 3, num_times))
    C = C.set(csdl.slice[0, 0, :], cp * cy)
    C = C.set(csdl.slice[0, 1, :], cp * sy)
    C = C.set(csdl.slice[0, 2, :], -sp)
    C = C.set(csdl.slice[1, 0, :], sr * sp * cy - cr * sy)
    C = C.set(csdl.slice[1, 1, :], sr * sp * sy + cr * cy)
    C = C.set(csdl.slice[1, 2, :], cp * sr)
    C = C.set(csdl.slice[2, 0, :], cr * sp * cy + sr * sy)
    C = C.set(csdl.slice[2, 1, :], cr * sp * sy - sr * cy)
    C = C.set(csdl.slice[2, 2, :], cp * cr)

    return C
# Derivative check and test of the body123_reference_frame_change function, using csdl.Recorder to track operations and check derivatives with respect to the input variables yaw, pitch, and roll, and printing the value of C at t=0 to verify it is the identity matrix when all angles are zero
if __name__ == "__main__":
    num_times = 100
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    yaw   = csdl.Variable(value=np.zeros(num_times), name='yaw')
    pitch = csdl.Variable(value=np.zeros(num_times), name='pitch')
    roll  = csdl.Variable(value=np.zeros(num_times), name='roll')

    C = body123_reference_frame_change(yaw, pitch, roll, num_times)

    recorder.stop()
    recorder.check_totals(of=C, wrt=[yaw, pitch, roll])
    print("C at t=0 (should be identity):\n", C.value[:, :, 0])