def update_thermo(particulator, T):
    env = particulator.env
    env['T'] = T
    env['a_w_ice'] = particulator.formulae.saturation_vapour_pressure.a_w_ice.py_func(T)


def run_simulation(particulator, temperature_profile, n_steps):
    output = {'products': {k: [] for k in particulator.products.keys()}, 'attributes': []}
    for step in range(n_steps+1):
        if step != 0:
            update_thermo(particulator, T=temperature_profile((step - .5) * particulator.dt))
            particulator.run(step - particulator.n_steps)
            update_thermo(particulator, T=temperature_profile(step * particulator.dt))
        else:
            output['spectrum'] = {}
            for k in ('n', 'freezing temperature', 'immersed surface area'):
                if k in particulator.attributes:
                    output['spectrum'][k] = particulator.attributes[k].to_ndarray()
        for k, v in particulator.products.items():
            output['products'][k].append(v.get()+0)
    return output
