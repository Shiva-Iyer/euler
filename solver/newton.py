# newton.py - Solve systems of equations using Newton's method
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

from numpy import array,dot
from numpy.linalg import norm
from gausseli import gausseli

def newton(f, dfdy, Y0, tol = 1E-12, maxiter = 20):
    for iter in range(maxiter):
        F = f(Y0)
        J = dfdy(Y0)
        Y = Y0 - gausseli(J, F)
        if (norm(Y - Y0, 2) <= tol):
            break

        Y0 = Y
    else:
        Y = array([])

    return(Y, iter + 1)
