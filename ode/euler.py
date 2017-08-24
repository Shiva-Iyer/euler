# euler.py - Solve ODE systems using the explicit Euler method
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

from numpy import linspace,zeros

def euler(f, a, b, n, Y0):
    h = (b - a)/(n - 1.0)
    t = linspace(a, b, n)

    Y = zeros([Y0.size,n])
    Y[:,[0]] = Y0[:,[0]].copy()

    for i in range(1, n):
        Y[:,[i]] = Y[:,[i-1]] + h*f(t[i-1], Y[:,[i-1]])

    return(t, Y)
