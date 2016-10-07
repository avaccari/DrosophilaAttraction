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
#from backgroundSubtractor import createBackgroundSubtractorRG
from backgroundSubtractor import createBackgroundSubtractorAVG


class userInt(object):
    def __init__(self):
        root = tk.Tk()
        root.withdraw()
        root.update()
        root.iconify()

    def chooseFile(self):
        fil = tkfd.askopenfilename()
        return fil

    def showInfo(self, txt):
        return tkmb.showinfo(title='INFO!',
                             message=txt,
                             icon=tkmb.INFO)

    def yesNo(self, txt):
        return tkmb.askyesno(title='YES/NO?',
                             message=txt,
                             icon=tkmb.QUESTION)


class main(object):
    def __init__(self, fil):
        # Instantiate user interface
        ui = userInt()

        if fil is None:
            fil = ui.chooseFile()

        if fil is '':
            raise IOError
        else:
            self.fil = fil

        self.mainWindow = basename(self.fil)
        cv2.namedWindow(self.mainWindow)

        self.frameHistoryLen = 30
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

#        self.fgbg = createBackgroundSubtractorRG(bufferSize=self.frameHistoryLen,
#                                                 display=False)
        self.fgbg = createBackgroundSubtractorAVG(bufferSize=self.frameHistoryLen,
                                                  alpha=0.05)

        self.sourceFrame = None
        self.processedFrame = None
        self.lastProcFrame = None
        self.workingFrame = None

        self.heatMap = None

        self.sumOpened = None

        self.pause = False

        self.selectionWindow = "Selection window"
        self.selPts = []
        self.selPtsNo = 0
        self.selectionMask = None
        self.selectionMode = False



    def processFrame(self):
        fg = self.fgbg.apply(self.sourceFrame, mask=False)

        self.processedFrame = fg

        return


#        if self.fgbg.isFullyInitialized() is True:
#            frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
#
#        self.processedFrame = frame.copy()

        # Opening to remove noise
        opened = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)

        self.processedFrame = opened

        return

        # Trace and heatmap
        if self.fgbg.isFullyInitialized() is True:
            if self.heatMap is None:
                self.heatMap = np.zeros_like(self.sourceFrame, dtype=np.float)
            self.heatMap += opened
            normalized = 255 * (self.heatMap / self.heatMap.max())
            _, mask = cv2.threshold(normalized.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
#            mask_inv = cv2.bitwise_not(mask)
#
#            source = cv2.bitwise_and(self.sourceFrame, self.sourceFrame, mask=mask_inv)
            src_alpha = cv2.cvtColor(self.sourceFrame, cv2.COLOR_GRAY2BGRA)

            colored = cv2.applyColorMap(normalized.astype(np.uint8),
                                        cv2.COLORMAP_HOT)
            colored = cv2.bitwise_and(colored, colored, mask=mask)
            b, g, r = cv2.split(colored)
            col_alpha = cv2.merge((b, g, r, normalized.astype(np.uint8)))

            self.processedFrame = cv2.add(src_alpha, col_alpha)
            self.lastProcFrame = self.processedFrame.copy()

            # Detect and draw contours
            source = opened.copy()
            im2, contours, hierarchy = cv2.findContours(source,
                                                        cv2.RETR_TREE,
                                                        cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(self.processedFrame, contours, -1, (0, 255, 0), 1)
        else:
            self.processedFrame = self.sourceFrame.copy()



    def showFrame(self, window, frame):
        if frame is not None:
            cv2.imshow(window, frame)
            cv2.waitKey(1)

    def readFrame(self, decode):
        frame = self.vid.readFrame(decode)

        if frame is None:
            return

        # Initialize frames for which we need to know the source size
        if self.sourceFrame is None:
            self.selectionMask = np.ones(frame.shape[:2], dtype=np.uint8)

        self.sourceFrame = frame.copy()

    def preprocessFrame(self):
        # Mask frame
        frame = cv2.bitwise_and(self.sourceFrame,
                                self.sourceFrame,
                                mask=self.selectionMask)

        # Pyramid down
        frame = cv2.pyrDown(frame, borderType=cv2.BORDER_REPLICATE)

        # Convert to gray
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Equalize historgram
        frame = cv2.equalizeHist(frame)

        self.sourceFrame = frame.copy()

    def selectRegions(self):
        # Pop-up a new window to select regions of interest
        cv2.namedWindow(self.selectionWindow)
        self.workingFrame = self.sourceFrame.copy()
        self.showFrame(self.selectionWindow, self.workingFrame)
        message = '--- Region selection process ---\n\n' + \
                  'Click around the boundaries of each region you wish to be analyzed. ' + \
                  'After 5 clicks, the points selected will be used to determine the ' + \
                  'enclosing elliptical region. If more regions are required, the process ' + \
                  'will continue. (q -> exit without selecting)'
        ui.showInfo(message)


        # Register mouse callbacks on selection window
        cv2.setMouseCallback(self.selectionWindow, self.mouseInteraction)
        self.selectionMode = True
        self.regions = []
        self.selPts = []
        self.selPtsNo = 0

        # Wait for user to be done
        while self.selectionMode:
            # Check for keypress
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.selectionMode = False

        cv2.destroyWindow(self.selectionWindow)


    def mouseInteraction(self, event, x, y, flags, params):
        # Left click
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.line(self.workingFrame, (x - 3, y), (x + 3, y), [0, 0, 255])
            cv2.line(self.workingFrame, (x, y - 3), (x, y + 3), [0, 0, 255])
            self.showFrame(self.selectionWindow, self.workingFrame)
            self.selPts.append([x, y])
            self.selPtsNo += 1

            # After 5 clicks draw ellipse
            if self.selPtsNo > 3:
                ellipse = cv2.fitEllipse(np.asarray(self.selPts))
                cv2.ellipse(self.workingFrame, ellipse, [0,255,0], 2)
                self.showFrame(self.selectionWindow, self.workingFrame)

                # Check with user
                message = 'Is this region ok?'
                ok = ui.yesNo(message)
                if ok is True:
                    self.regions.append(ellipse)

                self.workingFrame = self.sourceFrame.copy()
                self.showFrame(self.selectionWindow, self.workingFrame)
                self.selPts = []
                self.selPtsNo = 0

                message = 'Select another region?'
                ok = ui.yesNo(message)
                if ok is False:
                    self.selectionMode = False


    def watch(self, origFps):
        with video(self.fil) as self.vid:

            frameRate = int(self.vid.getFrameRate())

            if origFps is True:
                frameRate = 1

            frm = 0
            while self.vid.isFrameAvailable():
                # Check for keypress
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                process = False
                if frm % frameRate == 0:
                    process = True

                self.readFrame(process)

                if process is True:
                    if frm == 0:
                        self.selectRegions()
                    self.preprocessFrame()
                    self.processFrame()
                    self.showFrame(self.mainWindow, self.processedFrame)

                frm += 1

            cv2.waitKey(-1)

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        cv2.destroyWindow(self.mainWindow)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file",
                        help="File to analyze.")

    parser.add_argument("-o", "--original-fps",
                        dest='origFps',
                        default=False,
                        action='store_true',
                        help="Use original frame rate")

    args = parser.parse_args()

    again = True

    ui = userInt()

    while again is True:
        with main(args.file) as m:
            try:
                m.watch(args.origFps)
            except:
                pass

            # Do you want to analyze another file?
            again = ui.yesNo("Do you want to open another file?")
