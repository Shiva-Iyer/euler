# lineareq.py - Test the linear equation solvers
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

import sys
from os import path
from numpy import array

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from solver.gausseli import gausseli
from solver.tridiag import tdsolve
from solver.conjgrad import conjgrad

""" Solve the system x+y+z = 8, 2*x+3*y-4*z = -7, 3*x-5*y-7*z = -24
having the unique solution (3, 1, 4).
"""

A = array([[1.0, 1, 1], [2, 3, -4], [3, -5, -7]])
b = array([8.0, -7, -24])

x = gausseli(A, b)
print("Unique solution for invertible A matrix           : " + str(x))

""" Solve the system x+y+z = 8, 2*x+3*y-4*z = -7, 2*x+2*y+2*z = 16
having an infinite number of solutions, i.e. a consistent, 
underconstrained system.
"""

A = array([[1.0, 1, 1], [2, 3, -4], [2, 2, 2]])
b = array([8.0, -7, 16])

x = gausseli(A, b)
print("A solution for consistent, underconstrained system: " + str(x))

""" Solve the system x+y+z = 8, 3*x+3*y+3*z = 8, 3*x-5*y-7*z = -24
having no solutions, i.e. an inconsistent system.
"""

A = array([[1.0, 1, 1], [3, 3, 3], [3, -5, -7]])
b = array([8.0, 8, -24])

x = gausseli(A, b)
print("Solution for inconsistent system                  : " + str(x) + "\n")

""" Solve the tridiagonal system x-y = -5, 2x-y+3z = 0, y-6z = 1
having the unique solution (2, 7, 1).
"""

A = array([[1.0, -1.0, 0.0], [2.0, -1.0, 3.0], [0.0, 1.0, -6.0]])
b = array([-5.0, 0.0, 1.0])

x = tdsolve(A, b)
print("Solution for tridiagonal system                   : " + str(x))
print("")

""" Solve the symmetric positive-definite system below having unique
solution (1/2, 1/3, 1/5) using the conjugate gradient method
"""

A = array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
b = array([2.0/3.0, -1.0/30.0, 1.0/15.0])

x,iter = conjgrad(A, b)
print("Conjugate gradient soln (%3d iter) for SPD system : %s" % (iter, str(x)))
