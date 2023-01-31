import tal
print('Start read')
data = tal.io.read_capture('test/data/ZNLOS_Z_10_Single.hdf5')
print('Finish read')

tal.plot.xy_grid(data, size_x=8, size_y=4)

tal.plot.txy_interactive(data, slice_axis='t')
tal.plot.txy_interactive(data, slice_axis='x')
tal.plot.txy_interactive(data, slice_axis='y')
