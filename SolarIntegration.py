from numpy import interp, loadtxt


def solar_spectrum(solar, r):
    solar = loadtxt(solar)
    r_solar_t = 0
    sol_total = 0
    for i in range(len(r) - 1):
        if r[i, 0] > solar[0, 0] and r[i, 0] < solar[-1, 0]:
            wave = (r[i, 0] + r[i + 1, 0]) / 2
            p = interp(wave, solar[:, 0], solar[:, 1])
            r_solar_t += ((r[i, 1] + r[i + 1, 1]) / 2) * p * (r[i, 0] - r[i + 1, 0])
            sol_total += p * (r[i, 0] - r[i + 1, 0])

    r_solar = r_solar_t / sol_total
    return r_solar