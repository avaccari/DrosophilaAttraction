# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:15:13 2016
Name:    backgroundSubtractor.py
Purpose: Provides different background subtraction approaches
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
import matplotlib.pyplot as plt


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


class createBackgroundSubtractorAVG(object):
    def __init__(self, bufferSize=100, threshold=32, alpha=0.01):

        self.alpha = alpha
        self.bufferSize = bufferSize
        self.threshold = threshold

        self.fgbg = None

        self.cnt = 0

    def apply(self, frame, mask=False):
        self.cnt += 1

        if self.fgbg is None:
            self.fgbg = frame.astype(np.float, copy=True)

        bg = cv2.convertScaleAbs(self.fgbg)

        fg = cv2.subtract(frame, bg)

        _, fgmask = cv2.threshold(fg, self.threshold, 255, cv2.THRESH_BINARY)

        self.fgbg = cv2.accumulateWeighted(frame, self.fgbg, self.alpha)

        if mask is True:
            return fgmask, fg

        return fg

    def isFullyInitialized(self):
        return (self.cnt > self.bufferSize)


class createBackgroundSubtractorRG(object):
    def __init__(self, bufferSize=100, threshold=2.5, display=False):

        self.threshold = threshold
        self.bufferSize = bufferSize

        self.buffer = circBuffer(bufferSize)

        self.sizeBuffer = circBuffer(bufferSize)

        self.disp = None
        if display is True:
            self.disp = plot()

    def apply(self, frame, mask=False):
        fgmask = np.zeros_like(frame, dtype=np.bool)
        fgbg = np.zeros_like(frame, dtype=np.uint8)

        # If first time, initialize image buffer
        if self.buffer.getBuffer() is None:
            self.buffer.addElement(frame)
        else:
            # If image buffer is initialized perform detection
            if self.isImageBufferInitialized() is True:
                # Get Buffer and calculate average
                buff = self.buffer.getBuffer()
                mean = buff.mean(axis=-1)

                # Calculate difference
                diff = frame - mean

                # Evaluate outliers (foreground)
                var = buff.var(axis=-1)
                norm = (diff * diff) / var
                fgmask = norm > (self.threshold * self.threshold)
                fgSize = fgmask.sum()

                # If first detection, initialize FG size buffer
                if self.sizeBuffer.getBuffer() is None:
                    self.sizeBuffer.addElement(fgSize)
                else:
                    # If FG size buffer is initialized detect foreground
                    if self.isFullyInitialized() is True:
                        # Get FG size buffer and calculate average
                        buff = self.sizeBuffer.getBuffer()
                        sizeMean = buff.mean(axis=-1)

                        # Calculate difference
                        diff = fgSize - sizeMean

                        # Evaluate if outlier (size)
                        sizeVar = buff.var(axis=-1)
                        norm = (diff * diff) / sizeVar

                        # If size outlier, skip
                        if norm > 4 * (self.threshold * self.threshold):
                            if mask is True:
                                return fgmask, fgbg

                            return fgbg

                        # Process FG
                        fgbg = cv2.bitwise_and(frame, frame, mask=fgmask.astype(np.uint8))

                        # Replace the FG pixels with values drawn from the
                        # corresponding distribution. This will preserve
                        # the statistic in case the object in the
                        # foreground stops
                        fromStat = spsNorm.rvs(mean, np.sqrt(var), 1)
                        frame[fgmask] = fromStat[fgmask]

                    self.sizeBuffer.addElement(fgSize)

            self.buffer.addElement(frame)

        if self.disp is not None:
            self.disp.show(mean, var)

        if mask is True:
            return fgmask, fgbg

        return fgbg

    def isImageBufferInitialized(self):
        return self.buffer.isFullyInitialized()

    def isFullyInitialized(self):
        return self.sizeBuffer.isFullyInitialized()

