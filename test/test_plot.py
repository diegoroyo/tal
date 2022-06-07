import tal

print('Start read')
data = tal.io.read_capture('test/data/ZNLOS_Z_10_Single.hdf5')
print('Finish read')

tal.plot.xy_at_different_t(data, size_x=8, size_y=4)
