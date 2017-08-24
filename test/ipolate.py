# polinter.py - Test interpolation functions
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
from math import pi,sin,sqrt
from numpy import array

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from interpol import newton as newt
from interpol import spline as spln

""" Test interpolation on sin(x) where x is in degrees. X and Y are
the data points. Xint is the set of points to interpolate on.
"""

X = array([[0.0, 30.0, 45.0, 60.0, 90.0]]).T
Y = array([[sin(x*pi/180.0) for x in X]]).T

Xint = array([range(0, 100, 10)], dtype = "float64").T
Yexa = array([[sin(x*pi/180.0) for x in Xint]]).T

scheme = ["Newton divided differences", "Not-a-knot cubic spline",
          "Natural cubic spline", "Clamped cubic spline"]

for s in range(len(scheme)):
    if (s == 0):
        D = newt.ddtable(X, Y)
        Yint = newt.interp(D, X, Xint)
    elif (s >= 1 and s <= 3):
        S = spln.cspline(X, Y, s - 1)
        Yint = spln.interp(S, X, Xint)

    print("%s method:" % (scheme[s]))
    print("%-4s: %-8s %-8s %-12s" % (
        "x(o)", "Exact", "Interp.", "Error"))
    for i in range(Xint.size):
        print("%4.1f: %f %f %E" % (
            Xint[i], Yexa[i], Yint[i], abs(Yexa[i] - Yint[i])))

    print("")
