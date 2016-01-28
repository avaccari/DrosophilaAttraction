# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 07:29:54 2015
Name:    attraction.py
Purpose: Analyze larva motion in within a Pitri dish
Author:  Andrea Vaccari (av9g@virginia.edu)
Version: 0.0.0-alpha

    Copyright (C) Tue Nov 17 07:29:54 2015  Andrea Vaccari

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

# Add +-1 for axis normalization
# Add exclusion regions that shoud work as detection region as well
# User could select a spot and we create 1/10x1/10 of the +-1 range box around

import argparse
import Tkinter as tk
import tkMessageBox as tkmb
import tkFileDialog as tkfd
import cv2
from os.path import basename
import numpy as np
from cvVideo import video



class main(object):
    def __init__(self, fil):
        if fil is None:
            root = tk.Tk()
            root.withdraw()
            root.update()
            root.iconify()
            fil = tkfd.askopenfilename()

        if fil is '':
            raise IOError
        else:
            self.fil = fil

        self.mainWindow = basename(self.fil)
        cv2.namedWindow(self.mainWindow)

        self.sampleFactor = 1
        frameHistoryLen = 50
        self.frameHistory = frameHistoryLen * self.sampleFactor  # Should be larger than the detection history length
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=frameHistoryLen,
                                                       varThreshold=12,
                                                       detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self.sourceFrame = None
        self.previousFrame = None
        self.processedFrame = None
        self.lastProcFrame = None
        self.heatMap = None
        self.trace = None

        self.sumOpened = None

        self.detHistoryLen = 10
        self.detHistory = None
        self.detThres = 10000
        self.detMult = 10


    def processFrame(self):
        # Blur the image to deal with compression artifacts
        blurred = cv2.medianBlur(self.sourceFrame, 3)
#        blurred = self.sourceFrame

        # Detect foreground
        fore = self.fgbg.apply(blurred)

        # Opening to remove noise
        opened = cv2.morphologyEx(fore, cv2.MORPH_OPEN, self.kernel)
        self.sumOpened = np.sum(opened)

        # Initialize foreground detection history
        if self.detHistory is None:
            self.detHistory = [self.sumOpened] * self.detHistoryLen

        # Trace and heatmap
        validFrameNo = self.vid.getNextFrameNo() - self.frameHistory
        if validFrameNo > 0:
            # Check if fluke
            # Maybe should use IQR: https://en.wikipedia.org/wiki/Interquartile_range#Interquartile_range_and_outliers
            if self.sumOpened <= self.detMult * np.average(self.detHistory) or self.sumOpened <= self.detThres:
                self.detHistory.pop(0)
                self.detHistory.append(self.sumOpened)
                self.heatMap += opened
                normalized = 255 * (self.heatMap / self.heatMap.max())
                _, mask = cv2.threshold(normalized.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
    #            mask_inv = cv2.bitwise_not(mask)
    #
    #            source = cv2.bitwise_and(self.sourceFrame, self.sourceFrame, mask=mask_inv)
                src_alpha = cv2.cvtColor(self.sourceFrame, cv2.COLOR_BGR2BGRA)

                colored = cv2.applyColorMap(normalized.astype(np.uint8),
                                            cv2.COLORMAP_HOT)
                colored = cv2.bitwise_and(colored, colored, mask=mask)
                b, g, r = cv2.split(colored)
                col_alpha = cv2.merge((b, g, r, normalized.astype(np.uint8)))

                self.processedFrame = cv2.add(src_alpha, col_alpha)
                self.lastProcFrame = self.processedFrame.copy()
            else:
                self.processedFrame = self.lastProcFrame.copy()
        else:
            # Update foreground detection history before frame history count
            self.detHistory.pop(0)
            self.detHistory.append(self.sumOpened)
            self.processedFrame = self.sourceFrame.copy()

        # Detect and draw contours
        source = opened.copy()
        im2, contours, hierarchy = cv2.findContours(source,
                                                    cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.processedFrame, contours, -1, (0, 255, 0), 2)



    def annotateFrame(self):
        frameNo = self.vid.getNextFrameNo()

        if frameNo >= self.frameHistory:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.putText(self.processedFrame,
                    str(frameNo),
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color)

        text = str(self.detHistory) + ' --mean--> ' + str(np.average(self.detHistory))

        cv2.putText(self.processedFrame,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255))

        text = str(self.sumOpened) + ' <= ' + str(self.detThres)

        if self.sumOpened <= self.detThres:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.putText(self.processedFrame,
                    text,
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color)

        text =  str(self.sumOpened) + ' <= ' + str(self.detMult) +' x mean (=' + str(self.detMult * np.average(self.detHistory)) + ')'

        if self.sumOpened <= self.detMult * np.average(self.detHistory):
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.putText(self.processedFrame,
                    text,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color)


    def showFrame(self, window, frame):
        cv2.imshow(window, frame)


    def watch(self):
        with video(self.fil) as self.vid:

            templ= self.vid.getFrameTemplate()
            self.heatMap = np.zeros_like(templ, dtype=np.float32)[:,:,0]

            frm = 0;
            while self.vid.isFrameAvailable():
                process = False
                if frm % self.sampleFactor == 0:
                    process = True


                if self.sourceFrame is not None:
                    self.previousFrame = self.sourceFrame.copy()
                self.sourceFrame = self.vid.readFrame(process)

                if process:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break

                    self.processFrame()
                    self.annotateFrame()
                    self.showFrame(self.mainWindow, self.processedFrame)

                frm += 1

    def __enter__(self):
        return self


    def __exit__(self, exec_type, exec_value, traceback):
        cv2.waitKey(-1)
        cv2.destroyAllWindows()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file",
                        help="File to analyze.")

    args = parser.parse_args()

    again = True

    while again is True:
        with main(args.file) as m:
            m.watch()


        # Do you want to analyze another file?
        again = tkmb.askyesno("Analyze another?",
                              "Do you want to open another file?")
