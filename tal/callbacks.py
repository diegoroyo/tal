def plot_amplitude_phase(H_1, i, n):
    from tal.plot import amplitude_phase
    amplitude_phase(H_1, title=f'Step {i+1} out of {n}')
