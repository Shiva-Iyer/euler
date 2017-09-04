# polynom.py - Find the roots of polynomials using the Aberth method
# Copyright (C) 2017 Shiva Iyer <shiva.iyer AT g m a i l DOT c o m>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from math import cos,sin,pi
from numpy import abs,array,max,min,sum,zeros
from numpy.linalg import norm

def polyval(ply, x):
    val = 0.0
    for i in range(ply.size - 1):
        val = (val + ply[i,0]) * x

    val += ply[-1,0]

    return(val)

def polyder(ply, x):
    der = 0.0
    for i in range(ply.size - 2):
        der = (der + (ply.size - i - 1.0) * ply[i,0]) * x

    der += ply[-2,0]

    return(der)

def roots(ply, tol = 1E-3, maxiter = 20):
    n = ply.size
    a,b = abs(ply[n-1,0]), abs(ply[:n-1,0])
    lb = max([a / (a + max(b)), a / max([a, sum(b)])])
    a,b = abs(ply[0,0]), abs(ply[1:,0])
    ub = min([1.0 + max(b) / a, max([1.0, sum(b) / a])])

    rts = zeros([n-1,1], dtype = "complex")
    for i in range(0, n - 1, 2):
        mag = lb + (ub - lb) * (i + 1.0) / n
        pha = pi * (i + 1.0) / n
        if (i > n - 3):
            rts[i,0] = mag
        else:
            rts[i,0] = mag * (cos(pha) + 1.0j * sin(pha))
            rts[i+1,0] = rts[i,0].conj()

    cor = zeros([n-1,1], dtype = "complex")
    for iter in range(maxiter):
        for i in range(n - 1):
            rat = polyval(ply, rts[i,0]) / polyder(ply, rts[i,0])

            s = 0.0
            for j in range(n - 1):
                if (rts[i,0] != rts[j,0]):
                    s += 1.0 / (rts[i,0] - rts[j,0])

            cor[i,0] = rat / (rat * s - 1.0)

        rts += cor
        if (norm(cor, 2) <= tol):
            break
    else:
        rts = array([])

    return(rts, iter + 1)
