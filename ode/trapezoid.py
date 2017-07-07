# trapezoid.py - Solve ODE systems using the trapezoidal method
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

def trapezoid(f, dfdy, a, b, n, Y0):
    h = (b-a)/(n-1.0)
    I = eye(len(Y0))

    t = linspace(a, b, n)
    Y = zeros([n, len(Y0)])
    Y[0,:] = Y0
    for i in range(1, n):
        guess = Y0
	for iter in range(10):
	    b = Y[i-1,:] + h*(f(t[i-1], guess) + \
                f(t[i-1], Y[i-1,:]))/2.0 - guess
            if (norm(b, 2) <= 1E-12):
		Y[i,:] = guess
		break

	    J = h*dfdy(t[i-1], guess)/2.0 - I
	    guess = guess - gausseli(J, b)
	else:
            return(array([]), array([]))

    return(t, Y)
