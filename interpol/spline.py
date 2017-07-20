# spline.py - Interpolate using cubic splines
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
from solver.gausseli import gausseli
from solver.tridiag import tdsolve

def cspline(X, Y, type = 0, slope = array([0.0, 0.0])):
    if (not type in [0, 1, 2]):
        return(array([]))

    n = len(X)
    if (n != len(Y)):
        return(array([]))

    A = zeros([n, n])
    b = zeros(n)
    h = [X[i+1] - X[i] for i in range(n-1)]
    g = [Y[i+1] - Y[i] for i in range(n-1)]
    for i in range(1, n-1):
        A[i,i-1:i+2] = [h[i-1], 2.0*(h[i-1] + h[i]), h[i]]
        b[i] = 3.0*(g[i]/h[i] - g[i-1]/h[i-1])

    if (type == 0):    # "Not a knot" spline, the default
        A[0,:3] = [h[1], -h[0] - h[1], h[0]]
        A[n-1,n-3:n] = [h[n-2], -h[n-3] - h[n-2], h[n-3]]
    elif (type == 1):  # Natural spline
        A[0,0] = 1.0
        A[n-1,n-1] = 1.0
    else:              # Clamped spline
        A[0,:2] = [2*h[0], h[0]]
        A[n-1,n-2:n] = [h[n-2], 2*h[n-2]]
        b[0] = 3.0*(g[0]/h[0] - slope[0])
        b[n-1] = 3.0*(slope[1] - g[n-2]/h[n-2])

    if (type == 0):    # "Not a knot" spline
        c = gausseli(A, b)
    else:
        c = tdsolve(A, b)

    S = zeros([n - 1, 4])
    S[:,0] = Y[:n-1]
    S[:,2] = c[:n-1]
    for i in range(n-1):
        S[i,1] = (Y[i+1]-Y[i])/h[i] - h[i]*(2.0*c[i] + c[i+1])/3.0
        S[i,3] = (c[i+1] - c[i]) / (3.0*h[i])

    return(S)

def interp(S, X, Xint):
    m = len(Xint)
    n = len(X)

    Yint = zeros(m)
    for i in range(m):
        for j in range(n - 1):
            if (Xint[i] >= X[j] and Xint[i] < X[j+1]):
                r = j
                break
        else:
            if (Xint[i] < X[0]):
                r = 0
            else:
                r = n - 2

        x = Xint[i] - X[r]
        Yint[i] = S[r,0] + (S[r,1] + (S[r,2] + S[r,3]*x)*x)*x

    return(Yint)
