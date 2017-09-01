# impeuler.py - Solve ODE systems using the implicit Euler method
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

from numpy import array,eye,zeros
from numpy.linalg import norm
from linalg.gausseli import gausseli
from pde.jacobian import fdestim

def impeuler(f, dfdy, Y0, t):
    I = eye(Y0.size)
    Y = zeros([Y0.size,t.size])
    Y[:,[0]] = Y0[:,[0]].copy()

    for i in range(1, t.size):
        h = t[i] - t[i-1]
        guess = Y[:,[i-1]]
        for iter in range(10):
            c = Y[:,[i-1]] + h*f(t[i-1], guess) - guess
            if (norm(c, 2) <= 1E-12):
                Y[:,[i]] = guess[:,[0]]
                break

            if (not dfdy is None):
                J = h*dfdy(t[i-1], guess) - I
            else:
                J = h*fdestim(lambda z: f(t[i-1], z), guess, h) - I

            guess = guess - gausseli(J, c)
        else:
            return(array([]))

    return(Y)
