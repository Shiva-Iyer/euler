# curvefit.py - Test least squares curve fitting
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
from math import sqrt
from numpy import array

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from regress.leastsqr import linear,nonlin

"""
Find beta such that the curve T = beta*a^(3/2) minimizes the mean
square errors where a = semi-major axes of the solar system planets
in AU and T = their periods in years.
"""

sma = [0.3871, 0.7233, 1, 1.523, 5.205, 9.579, 19.20, 30.05, 39.24]
a_3_2 = array([a*sqrt(a) for a in sma])
T = array([0.2409, 0.6152, 1, 1.881, 11.86, 29.46, 84.01, 164.8, 247.7])

beta = linear(a_3_2, T)
print("Linear least squares curve fit             : beta = %6.4f" %
      (beta[0]))
print("")

"""
Find the non-linear curve fit y = beta1*x/(beta2+x) that minimizes the
mean square errors for the data points X and Y below.
"""

X = array([0.038, 0.194, 0.425, 0.6260, 1.2530, 2.5000, 3.7400])
Y = array([0.050, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317])

resf = lambda x, y, beta: (y-array([beta[0]*xi/(beta[1]+xi) for xi in x])).T
drdb = lambda x, y, beta: array([[-xi/(beta[1]+xi) for xi in x],
                                 [beta[0]*xi/(beta[1]+xi)**2 for xi in x]]).T

beta0 = array([0.1, 0.1])
beta,iter = nonlin(X, Y, beta0, resf, drdb)
print("Non-linear least squares curve fit (%1d iter): beta1 = %6.4f, beta2 = %6.4f" %
      (iter, beta[0], beta[1]))
