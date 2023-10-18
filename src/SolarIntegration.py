#  FOS: FOS, which means "light" in Greek, is used for Fast Optical Spectrum (FOS) calculations of nanoparticle media by combining Mie theory with either Monte Carlo simulations or a pre-trained deep neural network.
#  Copyright (C) 2023 Daniel Carne <dcarne@purdue.edu>
#  Copyright (C) 2023 Joseph Peoples <peoplesj@purdue.edu>
#  Copyright (C) 2023 Ziqi Guo <gziqi@purdue.edu>
#  Copyright (C) 2023 Dudong Feng <feng376@purdue.edu>
#  Copyright (C) 2023 Zherui Han <zrhan@purdue.edu>
#  Copyright (C) 2023 Xiulin Ruan <ruan@purdue.edu>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

from numpy import interp, ceil, zeros

def solar_spectrum(solar, r, t):
    # first find min and max wavelengths that both contain
    min_lambda = max(solar[0, 0], r[0, 0])
    max_lambda = min(solar[-1, 0], r[-1, 0])

    # find minimum wavelength delta
    d_lambda = min(min(solar[1:, 0] - solar[:-1, 0]), min(r[1:, 0] - r[:-1, 0]))

    # interpolate to match wavelengths and proper delta
    num_indices = int(ceil((max_lambda-min_lambda)/d_lambda))
    solar_interp = zeros((num_indices, 2))
    r_interp = zeros((num_indices, 2))
    t_interp = zeros((num_indices, 2))
    d_lambda = (max_lambda-min_lambda) / num_indices
    for i in range(num_indices):
        solar_interp[i, 0] = min_lambda + d_lambda*i
        r_interp[i, 0] = min_lambda + d_lambda * i
        t_interp[i, 0] = min_lambda + d_lambda * i
    solar_interp[:, 1] = interp(solar_interp[:, 0], solar[:, 0], solar[:, 1])
    solar_interp[:, 1] = interp(solar_interp[:, 0], solar[:, 0], solar[:, 1])
    r_interp[:, 1] = interp(r_interp[:, 0], r[:, 0], r[:, 1])
    t_interp[:, 1] = interp(t_interp[:, 0], t[:, 0], t[:, 1])

    # integrate values
    r_solar_t = 0
    t_solar_t = 0
    sol_total = 0
    for i in range(len(r_interp) - 1):
        p = (solar_interp[i, 1] + solar_interp[i+1, 1])/2
        r_solar_t += ((r_interp[i, 1] + r_interp[i + 1, 1]) / 2) * p * (r_interp[i, 0] - r_interp[i + 1, 0])
        t_solar_t += ((t_interp[i, 1] + t_interp[i + 1, 1]) / 2) * p * (t_interp[i, 0] - t_interp[i + 1, 0])
        sol_total += p * (r_interp[i, 0] - r_interp[i + 1, 0])
    r_solar = r_solar_t / sol_total
    t_solar = t_solar_t / sol_total
    a_solar = 1-t_solar-r_solar
    return r_solar, a_solar, t_solar