# qr.py - QR decomposition using Householder reflections
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

from cmath import exp,phase
from numpy import array,eye
from numpy.linalg import norm

def decomp(A):
    m,n = A.shape
    I = eye(m, dtype = A.dtype)
    Q = eye(m, dtype = A.dtype)

    for c in range(m-1):
        if (n != 1):
            x = Q.dot(A)[c:,[c]]
        else:
            x = Q.dot(A)[c:]

        a = norm(x, 2)
        if (a > 1E-12):
            a = -a * exp(1.0j*phase(x[0]))
            if (abs(a.imag) <= 1E-12):
                a = a.real
        else:
            continue

        u = x - a*I[c:,[c]]
        v = u / norm(u, 2)
        vct = v.conj().T

        Qc = I.copy()
        Qc[c:,c:] = I[c:,c:] - (1.0 + x.conj().T.dot(v) /
                                vct.dot(x)) * v.dot(vct)
        Q = Qc.dot(Q)

    R = Q.dot(A)
    Q = Q.conj().T

    return(Q, R)
