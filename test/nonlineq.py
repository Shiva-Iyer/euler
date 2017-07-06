# nonlineq.py - Test the non-linear equation solvers
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

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from numpy import array
from solver.newton import newton

""" Solve the system y^2 - x*y = 4, x^2 - xy = -3, which has solutions
(-3,-4) and (3,4).
"""

f = lambda y: array([y[1]*y[1]-y[0]*y[1]-4.0, y[0]*y[0]-y[0]*y[1]+3])
dfdy = lambda y: array([[-y[1], 2.0*y[1]-y[0]],
                        [2*y[0]-y[1], -y[0]]])

Y0 = array([1.0, 0])
Y,iter = newton(f, dfdy, Y0)
print("After %2d iterations from [1, 0]: x = %9.6f, y = %9.6f" % (
    iter, Y[0], Y[1]))

Y0 = array([0, 1.0])
Y,iter = newton(f, dfdy, Y0)
print("After %2d iterations from [0, 1]: x = %9.6f, y = %9.6f" % (
    iter, Y[0], Y[1]))
