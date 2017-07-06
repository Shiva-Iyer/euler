# gausseli.py - Solve systems of linear equations using Gauss elimination
#               with partial pivoting and back substitution
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

from numpy import array

def gausseli(A, b):
    if (A.ndim == 2 and b.ndim == 1):
        m,n = A.shape
        if (m != n or m != len(b)):
            return(array([]))
    else:
        return(array([]))

    for i in range(m):
        max = i
        for j in range(i + 1, m):
            if (abs(A[j,i]) > abs(A[max,i])):
                max = j

        if (max != i):
            A[i,:], A[max,:] = A[max,:].copy(), A[i,:].copy()
            b[i], b[max] = b[max], b[i]

        if (A[i,i] == 0.0):
            continue

        for j in range(i + 1, m):
            mul = A[j,i]/A[i,i]
            A[j,:] = A[j,:] - A[i,:]*mul
            b[j] -= b[i]*mul

    for i in range(m - 1, -1, -1):
        for j in range(i + 1, m):
            b[i] -= A[i,j]*b[j]
       
        if (A[i,i] != 0.0):
            b[i] /= A[i,i]
        else:
            if (b[i] == 0.0):
                b[i] = 1.0
            else:
                return(array([]))

    return(b)
