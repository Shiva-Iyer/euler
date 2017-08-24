# odesys.py - Test numerical schemes for ODE systems
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
from math import cos,exp,sin
from numpy import array
from numpy.linalg import norm

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from ode.euler import euler
from ode.impeuler import impeuler
from ode.trapezoid import trapezoid
from ode.rk4 import rk4
from ode.adams import adams
from ode.impadams import impadams

""" Solve the Initial Value Problem dy(t)/dt = A*y(t); y(0) = [1, 0]
where A = [0.2   -1]
          [  1  0.2]

The exact solution of this system is y(t) = [exp(0.2*t)*cos(t)]
                                            [exp(0.2*t)*sin(t)]
"""

e = 0.2
A = array([[e, -1], [1, e]])
Y0 = array([[1.0], [0]])

f = lambda t,Y: A.dot(Y)
dfdy = lambda t,Y: A

b = 1
n = 20 + 1

scheme = ["Euler", "Implicit Euler", "Trapezoidal",
          "4th order Runge-Kutta", "4-step Adams-Bashforth",
          "4-step Adams-Moulton"]
for i in range(len(scheme)):
    if (i == 0):
        t,Yap = euler(f, 0, b, n, Y0)
        Yex = array([[exp(e*x)*cos(x), exp(e*x)*sin(x)] for x in t]).T
    elif (i == 1):
        t,Yap = impeuler(f, dfdy, 0, b, n, Y0)
    elif (i == 2):
        t,Yap = trapezoid(f, dfdy, 0, b, n, Y0)
    elif (i == 3):
        t,Yap = rk4(f, 0, b, n, Y0)
    elif (i == 4):
        t,Yap = adams(f, 0, b, n, Y0, 4)
    elif (i == 5):
        t,Yap = impadams(f, None, 0, b, n, Y0, 3)

    print("%s method:" % scheme[i])
    print("%-5s %-17s %-17s %-12s" % ("Time", "Exact value",
                "Approximation", "Error norm"))
    for j in range(t.size):
        print("%4.2f: %f %f %f %f %E" % (t[j], Yex[0,j], Yex[1,j],
                Yap[0,j], Yap[1,j], norm(Yex[:,j]-Yap[:,j], 2)))

    print("")
