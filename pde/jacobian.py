# jacobian.py - Estimate the Jacobian matrix using finite differences
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

def fdestim(fy, Y, h):
    n = Y.size
    J = zeros([n,n])

    for dy in [-1.0, 1.0]:
        for i in range(n):
            Y1 = Y.copy()
            Y1[i,0] += dy*h
            fval = fy(Y1)
            J[:,[i]] += fval*dy

    J /= (2.0*h)

    return(J)
