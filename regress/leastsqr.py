# leastsqr.py - Regression analysis using least squares techniques
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

from numpy.linalg import norm
from linalg.cholesky import solve

def linear(X, y):
    C = X.conj().T
    return(solve(C.dot(X), C.dot(y)))

def nonlin(x, y, beta0, r, drdb, tol = 1E-6, maxiter = 10):
    beta = beta0.copy()

    for iter in range(maxiter):
        res = r(x, y, beta)
        Jr = drdb(x, y, beta)
        Jrct = Jr.conj().T

        db = solve(Jrct.dot(Jr), Jrct.dot(res))
        beta -= db
        if (norm(db, 2) <= tol):
            break

    return(beta, iter+1)
