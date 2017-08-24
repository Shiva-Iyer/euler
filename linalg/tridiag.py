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
    m,n = A.shape
    if (m != n or m != b.size):
        return(array([]))

    for i in range(n):
        for j in list(range(i-1)) + list(range(i+2, n)):
            if (A[i,j] != 0.0):
                return(array([]))

    c = zeros([n-1,1])
    c[0,0] = A[0,1] / A[0,0]
    for i in range(1, n-1):
        c[i,0] = A[i,i+1] / (A[i,i] - A[i,i-1]*c[i-1,0])

    d = zeros([n,1])
    d[0,0] = b[0,0] / A[0,0]
    for i in range(1, n):
        d[i,0] = (b[i,0]-A[i,i-1]*d[i-1,0])/(A[i,i]-A[i,i-1]*c[i-1,0])

    x = zeros([n,1])
    x[n-1,0] = d[n-1,0]
    for i in range(n-2, -1, -1):
        x[i,0] = d[i,0] - c[i,0]*x[i+1,0]

    return(x)
