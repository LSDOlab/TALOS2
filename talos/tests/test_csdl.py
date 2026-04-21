import csdl_alpha as csdl

recorder = csdl.Recorder(inline=True)
recorder.start()

x = csdl.Variable(value=1)
y = csdl.Variable(value=2)
z = x + y

recorder.stop()

print(z.value)