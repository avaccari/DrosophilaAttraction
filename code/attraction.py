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

# TODO:
# - Make sure the detection IS a larva
# - Handle the larva-target interaction
# - Metadata about regions
# - Store heatmap
# - If the larva is not detected, decide what to do with the various metrics
#   inf? NAN? 0? don't store it?


import argparse
import cv2
from os.path import basename, splitext
import numpy as np
import pandas as pd
#from backgroundSubtractor import createBackgroundSubtractorRG
from backgroundSubtractor import createBackgroundSubtractorAVG
from skimage import feature as skif
import sys
import time
import matplotlib
matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
import matplotlib.path as mplpath
from statsmodels.nonparametric.kde import KDEUnivariate

from userInt import userInt
from cvVideo import video


class larva(object):
    def __init__(self):
        self.contour = []
        self.center = None
        self.totalDist = 0
        self.lastDist = 0

    def updateContour(self, contour):
        self.contour = [contour]
        if self.contour:
            if self.center is None:
                self.center = contour.squeeze().mean(axis=0)
            center = contour.squeeze().mean(axis=0)
            trvld = self.center - center
            self.lastDist = np.sqrt(np.inner(trvld, trvld))
            self.totalDist += self.lastDist
            self.center = center

    def clearContour(self):
        self.contour = []



class region(object):
    def __init__(self, bbox, name=None):
        # The 2 factor is due to the reduction in size before processing should
        # probably be a global setting
        self.center = np.asarray(bbox[0]) / 2
        self.size = np.asarray(bbox[1]) / 2
        self.norm = self.size.max()
        self.angle = np.asarray(bbox[2])
        self.bbox = (self.center, self.size, self.angle)
        self.target = None
        self.name = name

        # Larvae are in regions
        self.larva = larva()  # What if more than one larva in more than one arena?

    def getDistances(self):
        d_ctr = np.inf
        d_trg = np.inf
        if self.larva.center is not None:
            larva_c = self.larva.center
            d_ctr = self.center - larva_c
            d_ctr = np.sqrt(np.inner(d_ctr, d_ctr))
            if self.target is not None:
                d_trg = self.target.center - larva_c
                d_trg = np.sqrt(np.inner(d_trg, d_trg))
        return d_ctr, d_trg

    def setTarget(self, bbox, name=None):
        self.target = region(bbox, name)


class main(object):
    def __init__(self, fil):
#        plt.ion()

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

        self.frameNo = 0
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

        # Create an empty pandas dataframe to store the data
        self.data = pd.DataFrame({'FrameNo': [],
                                  'FrameTimeMs': [],
                                  'LarvaPosX': [],
                                  'LarvaPosY': [],
                                  'LarvaLastDist': [],
                                  'LarvaLastVel': [],
                                  'LarvaTotalDist': [],
                                  'LarvaPosXNorm': [],
                                  'LarvaPosYNorm': [],
                                  'LarvaLastDistNorm': [],
                                  'LarvaLastVelNorm': [],
                                  'LarvaTotalDistNorm': [],
                                  'DistArenaCntrPxl': [],
                                  'DistTrgtCntrPxl': [],
                                  'DistArenaCntrNorm': [],
                                  'DistTrgtCntrNorm': []})

    def detectLarva(self, img):
        # Detect edges in the foreground and display
        edges = skif.canny(img, sigma=1.)
        edges_norm = cv2.normalize(edges.astype(np.uint8),
                                   None,
                                   0, 255,
                                   cv2.NORM_MINMAX)

        # Find countours in edges
        _, contours, _ = cv2.findContours(edges_norm,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_NONE)

        # If we found no contous, bail
        if not contours:
            return False

        # If we found at least one associate it to a larva and the correct arena
        # Might have to do some filtering here (area of larva ~ 30)
        larva_center = contours[0].squeeze().mean(axis=0)

        # Find the correct arena
        for arena in self.arenas:
            p = mplpath.Path(cv2.boxPoints(arena.bbox))
            if p.contains_point(larva_center):
                arena.larva.clearContour()
                arena.larva.updateContour(contours[0])

                # Decide how to handle the target because we only have the
                # bounding rectangle instead of the ellipse

        return True

    def preprocessFrame(self):
        # Pyramid down
        frame = cv2.pyrDown(self.sourceFrame, borderType=cv2.BORDER_REPLICATE)

        # Create mask first time around (hugly but necessary because of pyrDown)
        if self.selectionMask is None:
            self.selectionMask = np.ones_like(frame)
            if self.arenasNo != 0:
                self.selectionMask = np.zeros_like(frame)
                for a in self.arenas:
                    cv2.ellipse(self.selectionMask,
                                a.bbox,
                                [255, 255, 255],
                                -1)
            self.selectionMask = cv2.cvtColor(self.selectionMask,
                                              cv2.COLOR_BGR2GRAY)

        # Mask frame
        frame = cv2.bitwise_and(frame,
                                frame,
                                mask=self.selectionMask)

        # Convert to gray
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.sourceFrame = frame.copy()

    def processFrame(self):
        # Remove background (gets better with time)
        fg = self.fgbg.apply(self.sourceFrame, mask=False)

        # If background subtractor is initialized
        if self.fgbg.isFullyInitialized() is True:

            # Detect larva in the foreground
            if self.detectLarva(fg) is False:
                self.processedFrame = self.sourceFrame
                return

            # Create heatmap and add to original image in HOT colormap
            if self.heatMap is None:
                self.heatMap = np.zeros_like(self.sourceFrame, dtype=np.float)

            temp = np.zeros_like(self.heatMap)
            for arena in self.arenas:
                cv2.drawContours(temp, arena.larva.contour, 0, 1, cv2.FILLED)
            self.heatMap += temp
            heat_norm = cv2.normalize(self.heatMap/self.heatMap.max(),
                                      None,
                                      0, 255,
                                      cv2.NORM_MINMAX, cv2.CV_8UC1)
            heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_HOT)
            self.processedFrame = cv2.cvtColor(self.sourceFrame,
                                               cv2.COLOR_GRAY2BGR)
            cv2.add(self.processedFrame,
                    heat_color,
                    self.processedFrame,
                    mask=heat_norm)

            # Add center of selected arenas and their targets
            for arena in self.arenas:
                cv2.ellipse(self.processedFrame, arena.bbox, [255, 0, 0], 2)
                self.drawCross(self.processedFrame,
                               arena.center.astype(np.uint16),
                               color=[255, 0, 0])

                if arena.target is not None:
                    cv2.ellipse(self.processedFrame,
                                arena.target.bbox,
                                [0, 0, 255],
                                2)
                    self.drawCross(self.processedFrame,
                                   arena.target.center.astype(np.uint16))

                # Add larva contour to original image in green
                cv2.drawContours(self.processedFrame,
                                 arena.larva.contour,
                                 0,
                                 (0, 255, 0),
                                 1)
                if arena.larva.center is not None:
                    self.drawCross(self.processedFrame,
                                   arena.larva.center.astype(np.uint16),
                                   color=[0, 255, 0])

            # Store data
            self.storeData()

            # Annotate the frame
            self.annotateFrame()

        else:
            self.processedFrame = self.sourceFrame

    def storeData(self):
        arena = self.arenas[-1]
        larva = arena.larva
        norm = 1.0 / arena.norm
        time = self.vid.getMs() / 1000.0
        d_ctr, d_trg = arena.getDistances()
        l_pos = larva.center
        l_ldist = larva.lastDist
        try:
            lastDeltaFrame = self.frameNo - self.data.iloc[-1]['FrameNo']
        except IndexError:
            l_lvel = 0
        else:
            l_lvel = l_ldist / lastDeltaFrame
        l_tdist = larva.totalDist
        d_ctr_n = d_ctr * norm
        d_trg_n = d_trg * norm
        l_pos_n = l_pos * norm
        l_ldist_n = l_ldist * norm
        l_lvel_n = l_lvel * norm
        l_tdist_n = l_tdist * norm

        self.data = self.data.append({'FrameNo': self.frameNo,
                                      'FrameTimeMs': time,
                                      'LarvaPosX': l_pos[0],
                                      'LarvaPosY': l_pos[1],
                                      'LarvaLastDist': l_ldist,
                                      'LarvaLastVel': l_lvel,
                                      'LarvaTotalDist': l_tdist,
                                      'LarvaPosXNorm': l_pos_n[0],
                                      'LarvaPosYNorm': l_pos_n[1],
                                      'LarvaLastDistNorm': l_ldist_n,
                                      'LarvaLastVelNorm': l_lvel_n,
                                      'LarvaTotalDistNorm': l_tdist_n,
                                      'DistArenaCntrPxl': d_ctr,
                                      'DistTrgtCntrPxl': d_trg,
                                      'DistArenaCntrNorm': d_ctr_n,
                                      'DistTrgtCntrNorm': d_trg_n},
                                      ignore_index=True)


    def annotateFrame(self):
        txt = 'frame no: {0:5n}'.format(self.frameNo)

        cv2.putText(self.processedFrame,
                    txt,
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255))

    def showFrame(self, window, frame):
        if frame is not None:
            cv2.imshow(window, frame)
            cv2.waitKey(1)

    def readFrame(self, decode):
        frame = self.vid.readFrame(decode)

        if frame is None:
            return

        self.sourceFrame = frame.copy()

    def drawCross(self, window, center, color=[0, 0, 255], size=6):
        (x, y) = center
        sz = int(round(size / 2))
        cv2.line(window, (x - sz, y), (x + sz, y), color)
        cv2.line(window, (x, y - sz), (x, y + sz), color)

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

        cv2.destroyWindow(self.selectionWindow)

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
                center = np.asarray(ellipse[0]).astype(np.uint16)
                self.drawCross(self.workingFrame, center, [0, 255, 0], 10)
                self.showFrame(self.selectionWindow, self.workingFrame)

                # Check if region is ok
                message = 'Is this region ({0}) ok?'.format(self.selectionType)
                ok = ui.yesNo(message)
                if ok is True:
                    self.selPts = []
                    self.selPtsNo = 0

                    # If we are selecting an arena
                    if self.selectionType == 'Arena':
                        name = 'Arena-' + str(self.arenasNo)
                        self.arenas.append(region(ellipse, name))

                        # Check it we have a target
                        message = 'Add target for this arena?'
                        target = ui.yesNo(message)
                        if target is True:
                            self.selectionType = 'Target'
                        else:
                            self.arenasNo += 1
                    else:  # If we are defining the target
                        name = 'Target-' + str(self.arenasNo)
                        self.arenas[self.arenasNo].setTarget(ellipse, name)
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

    def processData(self):
        dst_arena = self.data['DistArenaCntrNorm'].values
        dst_trgt = self.data['DistTrgtCntrNorm'].values

        # Calculate statistics of distances
        dst_arena_avg = np.mean(dst_arena)
        dst_arena_med = np.median(dst_arena)
        dst_arena_stdev = np.std(dst_arena)
        dst_trgt_avg = np.mean(dst_trgt)
        dst_trgt_med = np.median(dst_trgt)
        dst_trgt_stdev = np.std(dst_trgt)

        # Create a statistics dataframe
        statData = pd.DataFrame({'DistArenaCntrAvg': dst_arena_avg,
                                 'DistArenaCntrMed': dst_arena_med,
                                 'DistArenaCntrStdev': dst_arena_stdev,
                                 'DistTrgtCntrAvg': dst_trgt_avg,
                                 'DistTrgtCntrMed': dst_trgt_med,
                                 'DistTrgtCntrStdev': dst_trgt_stdev},
                                 index=[0])

        # Evaluate KDE of normalized distance from center of arena
        d_arena_kde = KDEUnivariate(dst_arena)
        d_arena_kde.fit()

        # Evaluate KDE of normalized distance from center of target
        d_trgt_kde = KDEUnivariate(dst_trgt)
        d_trgt_kde.fit()

#        # Plot KDEs
#        plt.figure()
#        plt.plot(d_arena_kde.support, d_arena_kde.density, color='blue')
#        plt.plot(d_trgt_kde.support, d_trgt_kde.density, color='red')
#        plt.xlabel('Normalized distance')
#        plt.ylabel('Probability')
#        plt.title("KDE of distance from arena's (blue) and target's (red) centers")
#        plt.draw()
#        plt.show()

        # Create KDE dataframe
        kdeData = pd.DataFrame({'DistArenaCntrNorm': d_arena_kde.support,
                                'DistArenaCntrProb': d_arena_kde.density,
                                'DistTrgtCntrNorm': d_trgt_kde.support,
                                'DistTrgtCntrProb': d_trgt_kde.density})

        # Create assay metadata dataframe
        arena_size = self.arenas[-1].size * 2;
        arena_cntr = self.arenas[-1].center * 2;
        target_size = self.arenas[-1].target.size * 2;
        target_cntr = self.arenas[-1].target.center * 2;
        assyMetadata = pd.DataFrame({'ArenaWidth': arena_size[0],
                                     'ArenaHeight': arena_size[1],
                                     'ArenaCntrX': arena_cntr[0],
                                     'ArenaCntrY': arena_cntr[1],
                                     'TargerWidth': target_size[0],
                                     'TargetHeight': target_size[1],
                                     'TargetCntrX': target_cntr[0],
                                     'TargetCntrY': target_cntr[1]},
                                     index=[0])

        # Create and save Excel workbook
        writer = pd.ExcelWriter(splitext(self.fil)[0] + time.strftime('_%Y%m%d%H%M%S') + '.xlsx')
        self.data.to_excel(writer,
                           'Raw data',
                           columns=['FrameNo',
                                    'FrameTimeMs',
                                    'LarvaPosX',
                                    'LarvaPosY',
                                    'LarvaLastDist',
                                    'LarvaLastVel',
                                    'LarvaTotalDist',
                                    'LarvaPosXNorm',
                                    'LarvaPosYNorm',
                                    'LarvaLastDistNorm',
                                    'LarvaLastVelNorm',
                                    'LarvaTotalDistNorm',
                                    'DistArenaCntrPxl',
                                    'DistTrgtCntrPxl',
                                    'DistArenaCntrNorm',
                                    'DistTrgtCntrNorm'])
        kdeData.to_excel(writer,
                         'KDE data',
                         columns=['DistArenaCntrNorm',
                                  'DistArenaCntrProb',
                                  'DistTrgtCntrNorm',
                                  'DistTrgtCntrProb'])

        statData.to_excel(writer,
                          'Statistics',
                          columns=['DistArenaCntrAvg',
                                   'DistArenaCntrMed',
                                   'DistArenaCntrStdev',
                                   'DistTrgtCntrAvg',
                                   'DistTrgtCntrMed',
                                   'DistTrgtCntrStdev'])

        assyMetadata.to_excel(writer,
                              'Metadata',
                              columns=['ArenaWidth',
                                       'ArenaHeight',
                                       'ArenaCntrX',
                                       'ArenaCntrY',
                                       'TargerWidth',
                                       'TargetHeight',
                                       'TargetCntrX',
                                       'TargetCntrY'])

        writer.save()


    def watch(self, scaleFps):
        with video(self.fil) as self.vid:

            frameRate = 1

            if scaleFps is True:
                frameRate = int(self.vid.getFrameRate())

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
                if self.frameNo % frameRate == 0:
                    process = True

                self.readFrame(process)

                if process is True:
                    if self.frameNo == 0:
                        self.selectRegions()
                    self.preprocessFrame()
                    self.processFrame()
                    self.showFrame(self.mainWindow, self.processedFrame)

                self.frameNo += 1

            self.processData()

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
