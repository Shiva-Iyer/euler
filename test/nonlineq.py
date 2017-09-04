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
from numpy import array

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from solver.newton import newton
from solver.polynom import roots

""" Solve the system y^2 - x*y = 4, x^2 - xy = -3, which has solutions
(-3,-4) and (3,4).
"""

f = lambda y: array([[y[1,0]*y[1,0] - y[0,0]*y[1,0] - 4.0],
                     [y[0,0]*y[0,0] - y[0,0]*y[1,0] + 3.0]])

Y0 = array([[1.0], [0]])
Y,iter = newton(f, None, Y0)
print("Nonlin. sys: %2d iter. from [1, 0]   : x = %9.6f, y = %9.6f" % (
    iter, Y[0], Y[1]))

Y0 = array([[0], [1.0]])
Y,iter = newton(f, None, Y0)
print("Nonlin. sys: %2d iter. from [0, 1]   : x = %9.6f, y = %9.6f" % (
    iter, Y[0], Y[1]))

""" Solve the system x+y+z = 8, 2*x+3*y-4*z = -7, 3*x-5*y-7*z = -24
having the unique solution (3, 1, 4).
"""

A = array([[1.0, 1, 1], [2, 3, -4], [3, -5, -7]])
b = array([[8.0, -7, -24]]).T

f = lambda y: A.dot(y) - b
dfdy = lambda y: A

Y0 = array([[1.0, 1, 1]]).T
Y,iter = newton(f, dfdy, Y0)
print("")
print("Linear sys.: %2d iter. from [0, 0, 0]: x = %9.6f, y = %9.6f, z = %9.6f" % (
    iter, Y[0], Y[1], Y[2]))

"""
Find the roots of the polynomial x^3 + 3x^2 + 3x + 1 = 0 = (x + 1)^3
having the repeated root -1 with multiplicity 3.
"""

ply = array([[1.0, 3.0, 3.0, 1.0]]).T
rts,iter = roots(ply, tol = 1E-4, maxiter = 20)

print("")
print("Poly. roots:  %2d iterations: \n\t%s, \n\t%s, \n\t%s" % (
    iter, rts[0,0], rts[1,0], rts[2,0]))
