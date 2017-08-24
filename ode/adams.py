# adams.py - Solve ODE systems with the multistep Adams-Bashforth method
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

from numpy import array,linspace,zeros

_K = array([[1.0, 0, 0, 0, 0, 0],
            [1.5, -0.5, 0, 0, 0, 0],
            [x/12.0 for x in [23, -16, 5, 0, 0, 0]],
            [x/24.0 for x in [55, -59, 37, -9, 0, 0]],
            [x/720.0 for x in [1901, -2774, 2616, -1274, 251, 0]],
            [x/1440.0 for x in [4277, -7923, 9982, -7298, 2877, -475]]])

def adams(f, a, b, n, Y0, s = 2):
    if (s < 1 or s > 6):
        return(array([]), array([]))

    h = (b - a)/(n - 1.0)
    t = linspace(a, b, n)

    Y = zeros([Y0.size,n])
    Y[:,[0]] = Y0[:,[0]].copy()

    for i in range(1, n):
        if (i > s - 1):
            r = s - 1
        else:
            r = i - 1

        z = zeros([Y0.size,1])
        for j in range(r + 1):
            z += _K[r,j]*f(t[i-j-1], Y[:,[i-j-1]])

        Y[:,[i]] = (Y[:,[i-1]] + h*z)[:,[0]]

    return(t, Y)
