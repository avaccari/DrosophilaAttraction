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

import argparse
import Tkinter as tk
import tkMessageBox as tkmb
import tkFileDialog as tkfd
import cv2
from os.path import basename
import numpy as np
import matplotlib.pyplot as plt

class video(object):
    def __init__(self, vid):
        self.vid = vid

        self.frameNo = None
        self.totalFrames = None

        self.frameAvailable = None
        self.sourceFrame = None

        self.open()


    def open(self):
        self.cap = cv2.VideoCapture(self.vid)
        if self.cap.isOpened():
            self.frameNo = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.totalFrames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.frameAvailable = True
        else:
            raise IOError


    def readFrame(self):
        ret, frame = self.cap.read()

        if ret:
            self.frameNo = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.sourceFrame = frame
            return frame

        self.frameAvailable = False
        raise IOError


    def getFrameTemplate(self):
        template = None
        location = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

        currentFrame = None
        if self.sourceFrame is not None:
            currentFrame = self.sourceFrame.copy()

        try:
            template = self.readFrame()
        except IOError:
            pass

        self.rewind(location)

        if currentFrame is not None:
            self.sourceFrame = currentFrame.copy()

        return template


    def getFrameNo(self):
        return self.frameNo


    def isFrameAvailable(self):
        return self.frameAvailable


    def rewind(self, location=0.0):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, location)
        self.frameNo = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.sourceFrame = None


    def __enter__(self):
        return self


    def __exit__(self, exec_type, exec_value, traceback):
        self.cap.release()





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

        self.frameHistory = 50
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=self.frameHistory,
                                                       varThreshold=12,
                                                       detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self.sourceFrame = None
        self.previousFrame = None
        self.processedFrame = None
        self.heatMap = None
        self.trace = None


    def processFrame(self):
        # Blur the image to deal with compression artifacts
        blurred = cv2.medianBlur(self.sourceFrame, 3)

        # Detect foreground
        fore = self.fgbg.apply(blurred)


        # Opening to remove noise
        opened = cv2.morphologyEx(fore, cv2.MORPH_OPEN, self.kernel)

        # Detect contours
        source = opened.copy()
        im2, contours, hierarchy = cv2.findContours(source,
                                                    cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)

        # Trace and heatmap
        validFrameNo = self.vid.getFrameNo() - self.frameHistory
        if validFrameNo > 0:
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
        else:
            self.processedFrame = self.sourceFrame.copy()

        # Draw contours
        cv2.drawContours(self.processedFrame, contours, -1, (0, 255, 0), 2)



    def annotateFrame(self):
        frameNo = self.vid.getFrameNo()

        if frameNo >= self.frameHistory:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.putText(self.processedFrame,
                    str(frameNo),
                    (20,20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color)


    def showFrame(self, window, frame):
        cv2.imshow(window, frame)


    def watch(self):
        with video(self.fil) as self.vid:
            self.heatMap = np.zeros_like(self.vid.getFrameTemplate(),
                                         dtype=np.float32)[:,:,0]
            while self.vid.isFrameAvailable():
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                else:
                    try:
                        if self.sourceFrame is not None:
                            self.previousFrame = self.sourceFrame.copy()
                        self.sourceFrame = self.vid.readFrame()
                    except IOError:
                        pass
                    else:
                        self.processFrame()
                        self.annotateFrame()
                        self.showFrame(self.mainWindow, self.processedFrame)


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
        try:
            with main(args.file) as m:
                m.watch()
        except IOError:
            pass


        # Do you want to analyze another file?
        again = tkmb.askyesno("Analyze another?",
                              "Do you want to open another file?")
