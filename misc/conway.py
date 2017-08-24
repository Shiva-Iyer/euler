# conway.py - Function for modeling Conway's game of life
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

def life(I, gen, callback = None):
    m,n = I.shape

    C = I.copy()
    if (not callback is None):
        callback(0, C)

    P = zeros([m+2, n+2])
    P[1:m+1,1:n+1] = C.copy()

    for g in range(1, gen):
        for r in range(1, m+1):
            for c in range(1, n+1):
                nbr = P[r-1,c-1] + P[r-1,c] + P[r-1,c+1] + \
                      P[r,c-1] + P[r,c+1] + \
                      P[r+1,c-1] + P[r+1,c] + P[r+1,c+1]
                if (P[r,c] == 0):
                    if (nbr == 3):
                        C[r-1,c-1] = 1
                else:
                    if (nbr < 2 or nbr > 3):
                        C[r-1,c-1] = 0

        if (not callback is None):
            callback(g, C)

        P[1:m+1,1:n+1] = C.copy()

    return(C)
