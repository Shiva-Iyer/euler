# newton.py - Interpolate using divided differences and the Newton
#             form of the interpolating polynomial
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

from numpy import array,zeros

def ddtable(X, Y):
    n = len(X)
    if (n != len(Y)):
        return(array([]))

    D = zeros([n, n])
    D[:,0] = Y
    for c in range(1, n):
        for r in range(n-c):
            D[r,c] = (D[r+1,c-1] - D[r,c-1]) / (X[c+r] - X[r])

    return(D)

def interp(D, X, Xint):
    m = len(Xint)
    n = len(X)

    Yint = zeros(m)
    for i in range(m):
        for c in range(n):
            bas = array(D[0,c])
            for r in range(c):
                bas *= (Xint[i] - X[r])

            Yint[i] += bas

    return(Yint)
