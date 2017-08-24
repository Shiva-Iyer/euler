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
    m,n = A.shape
    if (m != n or m != b.size):
        return(array([]))

    C,x = A.copy(),b.copy()
    for i in range(m):
        max = i
        for j in range(i+1,m):
            if (abs(C[j,i]) > abs(C[max,i])):
                max = j

        if (max != i):
            C[i,:],C[max,:] = C[max,:].copy(),C[i,:].copy()
            x[i,0],x[max,0] = x[max,0],x[i,0]

        if (C[i,i] == 0.0):
            continue

        for j in range(i+1,m):
            if (C[j,i] != 0.0):
                mul = C[j,i]/C[i,i]
                C[j,:] = C[j,:] - C[i,:]*mul
                x[j,0] -= x[i,0]*mul

    for i in range(m-1,-1,-1):
        for j in range(i+1,m):
            x[i,0] -= C[i,j]*x[j,0]
       
        if (C[i,i] != 0.0):
            x[i,0] /= C[i,i]
        else:
            if (x[i,0] == 0.0):
                x[i,0] = 1.0
            else:
                return(array([]))

    return(x)
