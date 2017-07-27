# conjgrad.py - Solve linear equations with symmetric, positive-definite
#               matrices using the iterative conjugate gradient method
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

from numpy import array,zeros
from numpy.linalg import norm

def conjgrad(A, b, tol = 1E-12, maxiter = 10):
    if (A.ndim == 2 and b.ndim == 1):
        m,n = A.shape
        if (m != n or m != len(b)):
            return(array([]))
    else:
        return(array([]))

    x = zeros(n)
    r,p = b,b
    for iter in range(maxiter):
        Ap = A.dot(p)
        nr = norm(r, 2)**2

        alpha = nr / p.dot(Ap)
        x = x + alpha*p
        r = r - alpha*Ap
        nnr = norm(r, 2)
        beta = nnr**2 / nr
        p = r + beta*p

        if (nnr <= tol):
            break
    else:
        x = array([])

    return(x, iter + 1)
