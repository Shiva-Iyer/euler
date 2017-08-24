# cholesky.py - Cholesky decompose Hermitian positive-definite matrices
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

from math import sqrt
from numpy import array,eye,zeros

"""
If LDLform = False: Cholesky decompose Hermitian positive-definite A
such that A = LL*. If LDLform = True: decompose A such that A = LDL*
"""
def decomp(A, LDLform = False):
    m,n = A.shape
    if (m != n):
        return(array([]))

    if (LDLform):
        L = eye(n, dtype = A.dtype)
        D = zeros([n,n], dtype = A.dtype)
        for i in range(n):
            D[i,i] = A[i,i]
            for j in range(i):
                L[i,j] = A[i,j]
                for k in range(j):
                    L[i,j] -= L[i,k]*L[j,k].conj()*D[k,k]

                L[i,j] /= D[j,j]
                D[i,i] -= L[i,j]*L[i,j].conj()*D[j,j]

        return(L, D)
    else:
        L = zeros([n,n], dtype = A.dtype)
        for i in range(n):
            L[i,i] = A[i,i]
            for j in range(i):
                L[i,j] = A[i,j]
                for k in range(j):
                    L[i,j] -= L[i,k]*L[j,k].conj()

                L[i,j] /= L[j,j]
                L[i,i] -= L[i,j]*L[i,j].conj()

            L[i,i] = sqrt(L[i,i])

        return(L)

def solve(A, b):
    L,D = decomp(A, LDLform = True)
    if (L.size == 0):
        return(L)

    n = b.size
    x = b.copy()
    for i in range(n):
        for j in range(i):
            x[i,0] -= L[i,j]*x[j,0]

        x[i,0] /= L[i,i]

    L = D.dot(L.T)
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            x[i,0] -= L[i,j]*x[j,0]

        x[i,0] /= L[i,i]

    return(x)
