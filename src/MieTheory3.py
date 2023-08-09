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

from numpy import zeros, abs, sum, exp, conj, pi, ceil, sqrt, imag, real, complex128
from scipy.special import jv, yv, hankel1
from numba import njit


@njit()
def distribution(r, fv, dist):
    num_part = 101
    r1 = zeros(len(r) * num_part)
    fv1 = zeros(len(fv) * num_part)
    temp = zeros(num_part)
    half = (num_part - 1) / 2
    for i in range(len(r)):
        mean = r[i] * 2
        std = mean * dist / 2
        for j in range(num_part):
            var = abs(2 - 2 * (j / half))
            temp[j] = (1 / std) * exp(-0.5 * (var ** 2))
        temp_sum = sum(temp)

        for j in range(num_part):
            var = (-2 + 2 * (j / half))
            r1[i * num_part + j] = (mean - var * std) / 2

        for j in range(num_part):
            fv1[i * num_part + j] = (fv[i] * temp[j] / temp_sum)
    return r1, fv1


@njit()
def asy_calc(nmax, an, bn, cn, qs_i, j):

    p5 = zeros(int(nmax - 1))
    sumt = 0
    for k in range(int(nmax - 1)):
        p1 = (k + 1) * (k + 3) / (k + 2)
        p2t = (an[k] * conj(an[k + 1])) + (bn[k] * conj(bn[k + 1]))
        p2 = p2t.real
        p3 = (2 * (k + 1) + 1) / ((k + 1) * (k + 2))
        p4t = an[k] * conj(bn[k])
        p4 = p4t.real
        p5[k] = cn[k] * ((abs(an[k]) ** 2) + (abs(bn[k])) ** 2)
        sumt += p1 * p2 + p3 * p4
    asy_i_j = qs_i[j] * sumt / (0.5 * sum(p5))
    return asy_i_j


@njit()
def mie_coef(nmax, x, y, m_m, m_p, jx1, jx2, jx3, jy1, jy2, jy3, Yx1, Yx2, Yx3):
    an = zeros(int(nmax), dtype=complex128)
    bn = zeros(int(nmax), dtype=complex128)
    cn = zeros(int(nmax))


    H1 = complex(real(jx1)-imag(Yx1), imag(jx1)+real(Yx1))
    H2 = complex(real(jx2)-imag(Yx2), imag(jx2)+real(Yx2))
    H3 = complex(real(jx3)-imag(Yx3), imag(jx3)+real(Yx3))

    c1 = sqrt(pi * x / 2)
    c2 = 0.5 * sqrt(pi / (2 * y))
    c3 = 0.5 * sqrt(pi / (2 * x))
    for k in range(int(nmax)):

        any = c2 * (y * jy1 + jy2 - y * jy3) / (sqrt(pi * y / 2) * jy2)
        anx = c3 * (x * jx1 + jx2 - x * jx3) / (c1 * jx2)
        bnx = c3 * (x * H1 + H2 - x * H3) / (sqrt(pi * x / 2) * H2)
        psi_d_zetax = (jx2) / (H2)
        an[k] = psi_d_zetax * ((m_m * any - m_p * anx) / (m_m * any - m_p * bnx))
        bn[k] = psi_d_zetax * ((m_p * any - m_m * anx) / (m_p * any - m_m * bnx))
        cn[k] = 2 * (k + 1) + 1

        # update with recurrence formula
        jx1 = jx2
        jx2 = jx3
        jx3 = (2*(k+2.5)/x)*jx2-jx1
        jy1 = jy2
        jy2 = jy3
        jy3 = (2 * (k+2.5) / y) * jy2 - jy1

        Yx1 = Yx2
        Yx2 = Yx3
        Yx3 = (2 * (k + 2.5) / x) * Yx2 - Yx1
        H1 = complex(real(jx1) - imag(Yx1), imag(jx1) + real(Yx1))
        H2 = complex(real(jx2) - imag(Yx2), imag(jx2) + real(Yx2))
        H3 = complex(real(jx3) - imag(Yx3), imag(jx3) + real(Yx3))

    return an, bn, cn


@njit()
def main_loop(particles, paint, acr, wave, r1, fv1, jx1, jx2, jx3, jy1, jy2, jy3, Yx1, Yx2, Yx3, prop, thickness):
    qs_i = zeros(particles)
    qa_i = zeros(particles)
    asy_i = zeros(particles)
    for i in range(len(wave)):
        # for particle
        n_p = paint[i, 1]
        k_p = paint[i, 2]
        m_p = complex(n_p, k_p)

        # for matrix
        n_m = acr[i, 1]
        k_m = acr[i, 2]
        m_m = complex(n_m, k_m)

        for j in range(particles):
            r = r1[j]
            fv = fv1[j]

            x = 2 * pi * r * m_m / wave[i]
            y = 2 * pi * r * m_p / wave[i]
            nmax = ceil(abs(y) + 4.3 * abs(y) ** (1 / 3) + 2)

            an, bn, cn = mie_coef(nmax, x, y, m_m, m_p, jx1[i, j], jx2[i, j], jx3[i, j], jy1[i, j], jy2[i, j],
                                  jy3[i, j], Yx1[i, j], Yx2[i, j], Yx3[i, j])

            alpha = 4 * pi * r / wave[i] * k_m
            if k_m < (5 * 10 ** -7):
                gamma = 1
            else:
                gamma = 2 * (1 + (alpha - 1) * exp(alpha)) / (alpha ** 2)
            if gamma < 1:
                gamma = 1
            q2 = 0
            cext = 0
            for k in range(int(nmax)):
                q2 += cn[k] * (abs(an[k]) ** 2 + abs(bn[k]) ** 2)
                temp = (an[k] + bn[k]) / (m_m ** 2)
                cext += (wave[i] ** 2 / (2 * pi)) * (cn[k] * temp.real)
            csca = q2 * (wave[i] ** 2) * exp(-4 * pi * r * (m_m.imag) / wave[i]) / (2 * pi * gamma * abs(m_m) ** 2)
            qsca = csca / (pi * r * r)
            qext = cext / (pi * r * r)

            qs_i[j] = (1.5 * qsca * fv / (2 * r)) * 10 ** 4
            qt = (1.5 * qext * fv / (2 * r)) * 10 ** 4
            qa_i[j] = qt - qs_i[j]

            # compiled function for asymmetry parameter calculation
            asy_i[j] = asy_calc(nmax, an, bn, cn, qs_i, j)

        qs = sum(qs_i)
        qa = sum(qa_i)
        asy = sum(asy_i)
        # checking for bugs
        if qa < (10 ** -3):
            qa = 0
        if qs < (10 ** -3):
            qs = 0

        prop[0, i] = n_m
        prop[1, i] = qa
        prop[2, i] = qs
        prop[3, i] = asy
        prop[4, i] = thickness
    return prop



def mie_theory(r1, fv1, paint, acr, thickness, dist):

    if dist != 0:
        r1, fv1 = distribution(r1, fv1, dist)
    wave = paint[:, 0]
    prop = zeros((5, int(len(wave))))
    particles = int(len(r1))

    jx1 = zeros((len(wave), particles), dtype=complex128)
    jx2 = zeros((len(wave), particles), dtype=complex128)
    jx3 = zeros((len(wave), particles), dtype=complex128)
    jy1 = zeros((len(wave), particles), dtype=complex128)
    jy2 = zeros((len(wave), particles), dtype=complex128)
    jy3 = zeros((len(wave), particles), dtype=complex128)
    Yx1 = zeros((len(wave), particles), dtype=complex128)
    Yx2 = zeros((len(wave), particles), dtype=complex128)
    Yx3 = zeros((len(wave), particles), dtype=complex128)

    for i in range(len(wave)):
        for j in range(particles):
            # for particle
            n_p = paint[i, 1]
            k_p = paint[i, 2]
            m_p = complex(n_p, k_p)

            # for matrix
            n_m = acr[i, 1]
            k_m = acr[i, 2]
            m_m = complex(n_m, k_m)

            x = 2 * pi * r1[j] * m_m / wave[i]
            y = 2 * pi * r1[j] * m_p / wave[i]

            jx1[i, j] = jv(0.5, x)
            jx2[i, j] = jv(1.5, x)
            jx3[i, j] = (2 * 1.5 / x) * jx2[i, j] - jx1[i, j]
            jy1[i, j] = jv(0.5, y)
            jy2[i, j] = jv(1.5, y)
            jy3[i, j] = (2 * 1.5 / y) * jy2[i, j] - jy1[i, j]

            Yx1[i, j] = yv(0.5, x)
            Yx2[i, j] = yv(1.5, x)
            Yx3[i, j] = (2 * 1.5 / x) * Yx2[i, j] - Yx1[i, j]

    prop = main_loop(particles, paint, acr, wave, r1, fv1, jx1, jx2, jx3, jy1, jy2, jy3, Yx1, Yx2, Yx3, prop, thickness)

    return prop



def mie_theory_coreshell(r1, s1, fv1, paint_core, paint_shell, acr, thickness, dist):

    # r1 = core radius
    # s1 = shell thickness
    # fv1 = volume fraction of coreshell particles
        # r1, s1, fv1 = arrays of the same length
    # paint_core = core material refractive index
    # paint_shell = shell material refractive index
    # acr = matrix refractive index
    # thickness = thickness of the layer
    # dist = distribution of the particles (not used yet)   
        # maybe we can get distribution on r_total and then split it into r1 and s1


    r_total = r1 + s1 # total radius of the coreshell particle

    wave = paint_core[:, 0] # same as paint_shell[:, 0] and acr[:, 0]
    prop = zeros((5, int(len(wave))))
    particles = int(len(r1))


    qs_i = zeros(particles)
    qa_i = zeros(particles)
    asy_i = zeros(particles)

    for i in range(len(wave)):
        # for particle core
        n_core = paint_core[i, 1]
        k_core = paint_core[i, 2]
        m_core = complex(n_core, k_core)

        # for particle shell
        n_shell = paint_shell[i, 1]
        k_shell = paint_shell[i, 2]
        m_shell = complex(n_shell, k_shell)

        # for matrix
        n_m = acr[i, 1]
        k_m = acr[i, 2]
        m_m = complex(n_m, k_m)

        m1 = m_core/m_m
        m2 = m_shell/m_m
        m = m2/m1

        for j in range(particles):

            x = 2 * pi * r1[j] * m_m / wave[i] # size parameter for core
            y = 2 * pi * r_total[j] * m_m / wave[i]  # size paramete for shell


            nmax = ceil(abs(y) + 4.3 * abs(y) ** (1 / 3) + 2) # Converge criteria for bessel functions
            an = zeros(int(nmax), dtype=complex128)
            bn = zeros(int(nmax), dtype=complex128)
            cn = zeros(int(nmax))


            # calculate coreshell mie coefficients
            for n in range(1, int(nmax+1)):
                phiny = sqrt(pi*y/2)*jv(n+1/2, y)
                phiny2 = sqrt(pi*m2*y/2)*jv(n+1/2, m2*y)
                phinx = sqrt(pi*m2*x/2)*jv(n+1/2, m2*x)
                kany = -sqrt(pi*m2*y/2)*yv(n+1/2, m2*y)
                kanx = -sqrt(pi*m2*x/2)*yv(n+1/2, m2*x)
            #     kanx2 = -np.sqrt(np.pi*m1*x/2)*yv(n+1/2, m1*x)
                kanydy = -0.5*sqrt(pi/(2*m2*y))*(m2*y*yv(n-1/2, m2*y)
                                                        +yv(n+1/2, m2*y)
                                                        -m2*y*yv(n+3/2, m2*y))
                kanxdx = -0.5*sqrt(pi/(2*m2*x))*(m2*x*yv(n-1/2, m2*x)
                                                        +yv(n+1/2, m2*x)
                                                        -m2*x*yv(n+3/2, m2*x))
                ksen = sqrt(pi*y/2)*hankel1(n+1/2, y)
                Deny = 0.5*sqrt(pi/(2*m2*y))*(m2*y*jv(n-1/2, m2*y)
                                                    +jv(n+1/2, m2*y)
                                                    -m2*y*jv(n+3/2, m2*y)) \
                                                    /(sqrt(pi*m2*y/2)*jv(n+1/2, m2*y))
                Denx1 = 0.5*sqrt(pi/(2*m1*x))*(m1*x*jv(n-1/2, m1*x)
                                                        +jv(n+1/2, m1*x)
                                                        -m1*x*jv(n+3/2, m1*x)) \
                                                        /(sqrt(pi*m1*x)*jv(n+1/2, m1*x))
                Denx2 = 0.5*sqrt(pi/(2*m2*x))*(m2*x*jv(n-1/2, m2*x)
                                                        +jv(n+1/2, m2*x)
                                                        -m2*x*jv(n+3/2, m2*x)) \
                                                        /(sqrt(pi*m2*x)*jv(n+1/2, m2*x))

                phiny_m1 = sqrt(pi*y/2)*jv(n-1/2, y)
                ksen_m1 = sqrt(pi*y/2)*hankel1(n-1/2, y)

                An = phinx*(m*Denx1-Denx2)/(m*Denx1*kanx-kanxdx)
                Bn = phinx*(Denx1/m-Denx2)/(Denx1*kanx/m-kanxdx)

                Dn = (Deny-An*kanydy/phiny2)/(1-An*kany/phiny2)
                Gn = (Deny-Bn*kanydy/phiny2)/(1-Bn*kany/phiny2)


                an[n-1] = ((Dn/m2+n/y)*phiny-phiny_m1)/((Dn/m2+n/y)*ksen-ksen_m1)
                bn[n-1] = ((m2*Gn+n/y)*phiny-phiny_m1)/((m2*Gn+n/y)*ksen-ksen_m1)
                cn[n-1] = 2*n+1

            # calculate correction of matrix absorption
            alpha = 4 * pi * r_total / wave[i] * k_m
            if k_m < (5 * 10 ** -7):
                gamma = 1
            else:
                gamma = 2 * (1 + (alpha - 1) * exp(alpha)) / (alpha ** 2)
            if gamma < 1:
                gamma = 1

            # calculate scattering and extinction cross sections
            q2 = 0
            cext = 0
            for k in range(int(nmax)):
                q2 += cn[k] * (abs(an[k]) ** 2 + abs(bn[k]) ** 2)
                temp = (an[k] + bn[k]) / (m_m ** 2)
                cext += (wave[i] ** 2 / (2 * pi)) * (cn[k] * temp.real)
            csca = q2 * (wave[i] ** 2) * exp(-4 * pi * r_total * (m_m.imag) / wave[i]) / (2 * pi * gamma * abs(m_m) ** 2)
            qsca = csca / (pi * r_total * r_total)
            qext = cext / (pi * r_total * r_total)

            qs_i[j] = (1.5 * qsca * fv1[j] / (2 * r_total)) * 10 ** 4
            qt = (1.5 * qext * fv1[j] / (2 * r_total)) * 10 ** 4
            qa_i[j] = qt - qs_i[j]

            # compiled function for asymmetry parameter calculation
            asy_i[j] = asy_calc(nmax, an, bn, cn, qs_i, j)

        # summing up for each particle size
        qs = sum(qs_i)
        qa = sum(qa_i)
        asy = sum(asy_i)
        # checking for bugs
        if qa < (10 ** -3):
            qa = 0
        if qs < (10 ** -3):
            qs = 0

        prop[0, i] = n_m
        prop[1, i] = qa
        prop[2, i] = qs
        prop[3, i] = asy
        prop[4, i] = thickness


    return prop




@njit()
def effective_medium(optics_sum, vol_frac_sum, acr):
    wave = acr[:, 0]

    for i in range(len(optics_sum[0, :])):
        asy = optics_sum[3, i]

        k_m = acr[i, 2]
        qs = optics_sum[2, i]
        qa = optics_sum[1, i]
        if qs == 0:
            asy = 0
        else:
            asy = asy/qs
        if vol_frac_sum > 0.08:
            cor = 1 + 1.5 * vol_frac_sum - 0.75 * (vol_frac_sum ** 2)
            qs = qs * cor
            qa = 4 * pi * k_m * (10 ** 4) * (1 - vol_frac_sum) / wave[i] + qa * cor
        else:
            qa = 4 * pi * k_m * (10 ** 4) * (1 - vol_frac_sum) / wave[i] + qa

        # checking for bugs
        if qa < (10 ** -8):
            qa = 0
        if qs < (10 ** -8):
            qs = 0
        optics_sum[1, i] = qa
        optics_sum[2, i] = qs
        optics_sum[3, i] = asy
    return optics_sum