# polinter.py - Test polynomial interpolation functions
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

from interpol.newton import ddtable,interp

""" Test polynomial interpolation on sin(x) where x is in degrees.
X and Y are the data points used to construct the table of divided
differences. Xint is the set of interpolants.
"""

X = array([0.0, 30.0, 45.0, 60.0, 90.0])
Y = array([sin(x*pi/180.0) for x in X])

D = ddtable(X, Y)

Xint = array(range(0, 100, 10), dtype = "float64")
Yexa = array([sin(x*pi/180.0) for x in Xint])

Yint = interp(D, X, Xint)

print("Newton divided differences method:")
print("%-4s: %-8s %-8s %-12s" % (
    "x(o)", "Exact", "Interp.", "Error"))
for i in range(len(Xint)):
    print("%4.1f: %f %f %E" % (
        Xint[i], Yexa[i], Yint[i], abs(Yexa[i] - Yint[i])))
