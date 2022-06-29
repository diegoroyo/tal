import tal
from tal.plot import ByAxis
print('Start read')
data = tal.io.read_capture('test/data/ZNLOS_Z_10_Single.hdf5')
print('Finish read')


tal.plot.xy_grid(data, size_x=8, size_y=4)

tal.plot.txy_interactive(data, by = ByAxis.T)
tal.plot.txy_interactive(data, by = ByAxis.X)
tal.plot.txy_interactive(data, by = ByAxis.Y)

tal.plot.zxy_interactive(data.H, by = ByAxis.Z)
tal.plot.zxy_interactive(data.H, by = ByAxis.X)
tal.plot.zxy_interactive(data.H, by = ByAxis.Y)


