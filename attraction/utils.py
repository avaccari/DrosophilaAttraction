# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:06:50 2016
Name:    utils.py
Purpose: A set of utils used by various programs
Author:  Andrea Vaccari (av9g@virginia.edu)
Version: 0.0.0-alpha

    Copyright (C) Tue Mar  8 16:06:50 2016  Andrea Vaccari

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np


class circBuffer(object):
    """
    Circular buffer for arrays of any size.
    """
    def __init__(self, size):
        """
        Defines the size of the buffer.
        """
        self.size = size
        self.frameNo = 0
        self.buffer = None
        self.fullyInitialized = False
        self.totalFrames = 0

    def addElement(self, frame):
        """
        Adds the next element to the circular buffer

        The fist call allocates the entire buffer based on the shape of `frame`
        and the `size` of the buffer, and fills it with replicas of the first
        entry. Replicas are then overwritten as new elements are added.
        """
        if self.buffer is None:
            self.buffer = np.concatenate([frame[..., np.newaxis] for i in range(self.size)], axis=np.ndim(frame))
        else:
            self.buffer[..., self.frameNo] = frame

        self.totalFrames += 1
        self.frameNo = self.totalFrames % self.size

    def getBuffer(self):
        """
        Returns the entire buffer

        The buffer is returned as a large array stacked along an additional
        dimension respect to the buffer elements.

        Global operations can be performed along `axis=-1`.
        """
        return self.buffer

    def isFullyInitialized(self):
        """
        Return the initialization state of the buffer

        If `True` all elements in the buffer have been updated at least once.
        """
        return self.totalFrames >= self.size