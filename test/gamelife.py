# gamelife.py - Test Conway's game of life simulator
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

from misc.conway import life

def dispgen(gen, C):
    print(C)

# Simulate the beacon oscillator having a period of 2 generations
I = array([[0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0],
           [0, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 0],
           [0, 0, 0, 1, 1, 0],
           [0, 0, 0, 0, 0, 0]])

life(I, 8, dispgen)
