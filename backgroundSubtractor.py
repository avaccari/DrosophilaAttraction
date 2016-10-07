# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:15:13 2016
Name:
Purpose:
Author:  Andrea Vaccari (av9g@virginia.edu)
Version: 0.0.0-alpha

    Copyright (C) Thu Mar 31 11:15:13 2016  Andrea Vaccari

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

import cv2
import numpy as np
from utils import circBuffer
from scipy.stats import norm as spsNorm


class plot(object):
    def __init__(self):
        cv2.namedWindow('Mean')
        cv2.namedWindow('Stdev')
        cv2.namedWindow('SNR')

    def __del__(self):
        cv2.destroyWindow('Mean')
        cv2.destroyWindow('Stdev')
        cv2.destroyWindow('SNR')

    def show(self, mean, var):
        stdev = np.sqrt(var)
        snr = 255 * (1 - mean / (mean + stdev))

        mean = mean.astype(np.uint8)
        stdev = stdev.astype(np.uint8)
        snr = snr.astype(np.uint8)

        stdev = cv2.applyColorMap(stdev, cv2.COLORMAP_JET)
        snr = cv2.applyColorMap(snr, cv2.COLORMAP_JET)

        cv2.imshow('Mean', mean)
        cv2.imshow('Stdev', stdev)
        cv2.imshow('SNR', snr)

        cv2.waitKey(1)


class createBackgroundSubtractorRG(object):
    def __init__(self, bufferSize=100, threshold=2.5, display=False):

        self.threshold = threshold
        self.bufferSize = bufferSize

        self.frameNo = 0
        self.buffer = circBuffer(bufferSize)

        self.disp = None
        if display is True:
            self.disp = plot()

    def apply(self, frame, mask=False):
        if self.buffer.getBuffer() is None:
            self.buffer.addElement(frame)
            mask = np.ones_like(frame, dtype=np.bool)
            fgbg = frame
            self.mean = frame
            self.var = np.zeros_like(frame)
        else:
            buff = self.buffer.getBuffer()

            self.mean = np.mean(buff, axis=-1)
            diff = frame - self.mean
            self.var = np.var(buff, axis=-1)
            norm = (diff * diff) / self.var
            mask = norm > (self.threshold * self.threshold)

            fgbg = cv2.bitwise_and(frame, frame, mask=mask.astype(np.uint8))

            # If the buffer is fully initialized, replace the foreground pixels
            # with values drawn from the corresponding distribution. This will
            # preserve the statistic in case the object in the foreground stops
            if self.isFullyInitialized() is True:
                fromStat = spsNorm.rvs(self.mean, np.sqrt(self.var), 1)
                frame[mask] = fromStat[mask]

            self.buffer.addElement(frame)

        self.frameNo += 1

        if self.disp is not None:
            self.disp.show(self.mean, self.var)

        if mask is True:
            return mask, fgbg

        return fgbg

    def isFullyInitialized(self):
        return self.buffer.isFullyInitialized()

