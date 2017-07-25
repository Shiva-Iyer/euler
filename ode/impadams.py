# impadams.py - Solve ODEs using the multistep Adams-Moulton method
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

from numpy import array,eye,linspace,zeros
from numpy.linalg import norm
from solver.gausseli import gausseli
from pde.jacobian import fdestim

_K = array([[1.0, 0, 0, 0, 0, 0],
            [0.5, 0.5, 0, 0, 0, 0],
            [x/12.0 for x in [5, 8, -1, 0, 0, 0]],
            [x/24.0 for x in [9, 19, -5, 1, 0, 0]],
            [x/720.0 for x in [251, 646, -264, 106, -19, 0]],
            [x/1440.0 for x in [475, 1427, -798, 482, -173, 27]]])

def impadams(f, dfdy, a, b, n, Y0, s = 2):
    if (s < 0 or s > 5):
        return(array([]), array([]))

    h = (b - a)/(n - 1.0)
    I = eye(len(Y0))

    t = linspace(a, b, n)
    Y = zeros([n, len(Y0)])
    Y[0,:] = Y0

    for i in range(1, n):
        if (i >= s):
            r = s
        else:
            r = i

        guess = Y[i-1,:]
	for iter in range(10):
            c = _K[r,0]*f(t[i-1], guess)
            for j in range(1, r+1):
                c += _K[r,j]*f(t[i-j], Y[i-j])

            c = Y[i-1,:] + h*c - guess
            if (norm(c, 2) <= 1E-12):
		Y[i,:] = guess
		break

            if (not dfdy is None):
                J = h*_K[r,0]*dfdy(t[i-1], guess) - I
            else:
                J = h*_K[r,0]*fdestim(lambda z: f(t[i-1],z),guess,h) - I

            guess = guess - gausseli(J, c)
        else:
            return(array([]), array([]))

    return(t, Y)
