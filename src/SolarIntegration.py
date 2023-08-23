#  FOS: FOS, which means "light" in Greek, is used for Fast Optical Spectrum (FOS) calculations of nanoparticle media by combining Mie theory with either Monte Carlo simulations or a pre-trained deep neural network.
#  Copyright (C) 2023 Daniel Carne <dcarne@purdue.edu>
#  Copyright (C) 2023 Joseph Peoples <@gmail.com>
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

from numpy import interp

def solar_spectrum(solar, r, t):
    r_solar_t = 0
    t_solar_t = 0
    sol_total = 0
    solar_2 = r.copy()
    solar_2[:, 1] = 0
    solar_2[:, 1] = interp(solar_2[:, 0], solar[:, 0], solar[:, 1])

    for i in range(len(solar_2[:, 0])):
        if solar_2[i, 0] < solar[0, 0] or solar_2[i, 0] > solar[-1, 0]:
            solar_2[i, 1] = 0

    for i in range(len(r) - 1):
        p = (solar_2[i, 1] + solar_2[i+1, 1])/2
        r_solar_t += ((r[i, 1] + r[i + 1, 1]) / 2) * p * (r[i, 0] - r[i + 1, 0])
        t_solar_t += ((t[i, 1] + t[i + 1, 1]) / 2) * p * (t[i, 0] - t[i + 1, 0])
        sol_total += p * (r[i, 0] - r[i + 1, 0])

    r_solar = r_solar_t / sol_total
    t_solar = t_solar_t / sol_total
    a_solar = 1-t_solar-r_solar
    return r_solar, a_solar, t_solar

