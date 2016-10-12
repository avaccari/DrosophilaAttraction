# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 07:29:54 2015
Name:    attraction.py
Purpose: Analyze larva motion within a Pitri dish
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
import cv2
from os.path import basename
import numpy as np
from cvVideo import video
#from backgroundSubtractor import createBackgroundSubtractorRG
from backgroundSubtractor import createBackgroundSubtractorAVG
from skimage import measure as skim
from skimage import feature as skif
from skimage import segmentation as skis
import sys
import matplotlib.pyplot as plt
from userInt import userInt


class region(object):
    def __init__(self, bbox):
        self.bbox = bbox
        self.center = np.asarray(bbox[0])
        self.size = np.asarray(bbox[1])
        self.angle = np.asarray(bbox[2])
        self.target = None

    def setTarget(self, bbox):
        self.target = region(bbox)



class main(object):
    def __init__(self, fil):
        # Instantiate user interface
        ui = userInt()

        if fil is None:
            fil = ui.chooseFile()

        if fil is '':
            raise IOError('File opening cancelled by user')
        else:
            self.fil = fil

        self.mainWindow = basename(self.fil)
        cv2.namedWindow(self.mainWindow)

        self.frameHistoryLen = 50
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        self.fgbg = createBackgroundSubtractorAVG(bufferSize=self.frameHistoryLen,
                                                  alpha=0.02)

        self.sourceFrame = None
        self.processedFrame = None
        self.lastProcFrame = None
        self.workingFrame = None

        self.heatMap = None

        self.selectionWindow = 'Selection Window'
        self.arenas = []
        self.arenasNo = 0
        self.selPts = []
        self.selPtsNo = 0
        self.selectionMask = None
        self.selectionMode = False
        self.selectionType = None


    def detectLarva(self, img):
        # Detect edges in the foreground and display
        larva = skif.canny(img, sigma=1.)
        larva_norm = cv2.normalize(larva.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)

        # Find countours in edges
        _, contours, hierarchy = cv2.findContours(larva_norm,
                                                  cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_NONE)

        # Might have to do some filtering here (area of larva ~ 30)
                
        return contours
        
        
    def processFrame(self):
        # Remove background (gets better with time)
        fg = self.fgbg.apply(self.sourceFrame, mask=False)

        # If background subtractor is initialized
        if self.fgbg.isFullyInitialized() is True:
            # Show detected foreground
#            self.showFrame('Foreground', fg)

            # Detect larva in the foreground
            contours = self.detectLarva(fg)            

            # Add detection to original image in green
            self.processedFrame = cv2.cvtColor(self.sourceFrame, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(self.processedFrame, contours, 0, (0, 255, 0), 1)
    
            # Create heatmap from detection
            if self.heatMap is None:
                self.heatMap = np.zeros_like(self.sourceFrame, dtype=np.float)

            larva = np.zeros_like(self.heatMap)
            cv2.drawContours(larva, contours, 0, 1, cv2.FILLED)
            self.heatMap += larva
            heat_norm = cv2.normalize(self.heatMap/self.heatMap.max(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_HOT)
            self.showFrame('Heatmap', heat_color)

        else:
            self.processedFrame = self.sourceFrame

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

        self.sourceFrame = frame.copy()


    def selectRegions(self):
        # Pop-up a new window to select regions of interest
        cv2.namedWindow(self.selectionWindow)
        self.workingFrame = self.sourceFrame.copy()
        self.showFrame(self.selectionWindow, self.workingFrame)
        message = '--- Regions selection process ---\n\n' + \
                  'Click around the boundaries of each region you wish to be analyzed. ' + \
                  'After 5 clicks, the points selected will be used to determine the ' + \
                  'enclosing elliptical arena. After selecting each arena, you will be ' + \
                  'asked to select the corresponding target (if any) with the same procedure. ' + \
                  'The procedure can be repeated for all the required arenas.\n\n' + \
                  '(q -> exit without selecting)'
        ui.showInfo(message)

        # Register mouse callbacks on selection window
        cv2.setMouseCallback(self.selectionWindow, self.mouseInteraction)
        self.selectionMode = True
        self.selectionType = 'Arena'

        # Select arena
        while self.selectionMode:
            # Check for keypress
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.regions = []
                self.selectionMode = False

        # Create mask based on selections
        if self.arenasNo == 0:
            self.selectionMask = np.ones_like(self.workingFrame)
        else:
            self.selectionMask = np.zeros_like(self.workingFrame)
            for a in self.arenas:
                cv2.ellipse(self.selectionMask, a.bbox, [255, 255, 255], -1)
        self.selectionMask = cv2.cvtColor(self.selectionMask, cv2.COLOR_BGR2GRAY)

        cv2.destroyWindow(self.selectionWindow)


    def drawCross(self, window, center, color=[0, 0, 255], size=6):
        (x, y) = center
        sz = int(round(size / 2))
        cv2.line(window, (x - sz, y), (x + sz, y), color)
        cv2.line(window, (x, y - sz), (x, y + sz), color)


    def mouseInteraction(self, event, x, y, flags, params):
        # Left click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawCross(self.workingFrame, (x, y))
            self.showFrame(self.selectionWindow, self.workingFrame)
            self.selPts.append([x, y])
            self.selPtsNo += 1

            # After 5 clicks draw ellipse and its center
            if self.selPtsNo > 4:
                ellipse = cv2.fitEllipse(np.asarray(self.selPts))
                cv2.ellipse(self.workingFrame, ellipse, [0, 255, 0], 2)
                center = np.asarray(ellipse[0]).astype(np.int)
                self.drawCross(self.workingFrame, center, [0, 255, 0], 10)
                self.showFrame(self.selectionWindow, self.workingFrame)

                # Check if region is ok
                message = 'Is this region ({0}) ok?'.format(self.selectionType)
                ok = ui.yesNo(message)
                if ok is True:
                    self.selPts = []
                    self.selPtsNo = 0

                    if self.selectionType == 'Arena':
                        self.arenas.append(region(ellipse))

                        # Check it we have a target
                        message = 'Add target for this arena?'
                        target = ui.yesNo(message)
                        if target is True:
                            self.selectionType = 'Target'
                        else:
                            self.arenasNo += 1
                    else:  # If we are defining the target
                        self.arenas[self.arenasNo].setTarget(ellipse)
                        self.arenasNo += 1
                        self.selectionType = 'Arena'

                self.workingFrame = self.sourceFrame.copy()
                self.showFrame(self.selectionWindow, self.workingFrame)
                self.selPts = []
                self.selPtsNo = 0

                message = 'Select another region ({0})?'.format(self.selectionType)
                ok = ui.yesNo(message)
                if ok is False:
                    self.selectionMode = False



    def annotateFrame(self):
        cv2.putText(self.processedFrame,
                    str(int(self.vid.getMs()/1000)) + 's',
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    255)


    def watch(self, scaleFps):
        with video(self.fil) as self.vid:

            frameRate = 1

            if scaleFps is True:
                frameRate = int(self.vid.getFrameRate())

            frm = 0
            while self.vid.isFrameAvailable():
                # Check for keypress
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(-1)
                elif key == ord('h'):
                    message = '--- Main help ---\n\n' + \
                              'h -> show this help\n' + \
                              'p -> toggle pause mode\n' + \
                              'q -> quit the current analysis\n'
                    ui.showInfo(message)

                process = False
                if frm % frameRate == 0:
                    process = True

                self.readFrame(process)

                if process is True:
                    if frm == 0:
                        self.selectRegions()
                    self.preprocessFrame()
                    self.processFrame()
                    self.annotateFrame()
                    self.showFrame(self.mainWindow, self.processedFrame)

                frm += 1


    def __enter__(self):
        return self


    def __exit__(self, exec_type, exec_value, traceback):
        cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file",
                        help="File to analyze.")

    parser.add_argument("-s", "--scale-fps",
                        dest='scaleFps',
                        default=False,
                        action='store_true',
                        help="Scale frame rate")

    args = parser.parse_args()

    again = True

    ui = userInt()

    while again is True:
        with main(args.file) as m:
            try:
                m.watch(args.scaleFps)
            except:
                print sys.exc_info()

            # Do you want to analyze another file?
            again = ui.yesNo("Do you want to open another file?")
