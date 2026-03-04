import csdl_alpha as csdl
import numpy as np

recorder = csdl.Recorder(inline=True)
recorder.start()
f = csdl.Variable(value=np.zeros((3,3)))
f = f.set(csdl.slice[0, 1], csdl.Variable(value=5.0))
recorder.stop()
print(f.value)