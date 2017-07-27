# tridiag.py - Solve tridiagonal systems of equations using the
#              Thomas tridiagonal matrix algorithm
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

def tdsolve(A, b):
    if (A.ndim == 2 and b.ndim == 1):
        m,n = A.shape
        if (m != n or m != len(b)):
            return(array([]))

        for i in xrange(n):
            for j in range(i-1) + range(i+2, n):
                if (A[i,j] != 0.0):
                    return(array([]))
    else:
        return(array([]))

    c = zeros(n-1)
    c[0] = A[0,1] / A[0,0]
    for i in xrange(1, n-1):
        c[i] = A[i,i+1] / (A[i,i] - A[i,i-1]*c[i-1])

    d = zeros(n)
    d[0] = b[0] / A[0,0]
    for i in xrange(1, n):
        d[i] = (b[i] - A[i,i-1]*d[i-1]) / (A[i,i] - A[i,i-1]*c[i-1])

    x = zeros(n)
    x[n-1] = d[n-1]
    for i in xrange(n-2, -1, -1):
        x[i] = d[i] - c[i]*x[i+1]

    return(x)
