# rk4.py - Solve ODE systems using the 4th order Runge-Kutta method
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

def rk4(f, a, b, n, Y0):
    h = (b - a)/(n - 1.0)
    t = linspace(a, b, n)

    Y = zeros([Y0.size,n])
    Y[:,[0]] = Y0[:,[0]].copy()

    for i in range(1, n):
        k1 = h*f(t[i-1], Y[:,[i-1]])
        k2 = h*f(t[i-1] + 0.5*h, Y[:,[i-1]] + 0.5*k1)
        k3 = h*f(t[i-1] + 0.5*h, Y[:,[i-1]] + 0.5*k2)
        k4 = h*f(t[i-1] + h, Y[:,[i-1]] + k3)
        Y[:,[i]] = (Y[:,[i-1]] + (k1 + k4)/6.0 + (k2 + k3)/3.0)[:,[0]]

    return(t, Y)
