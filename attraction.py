#!/usr/local/bin/python
"""
Created on Tue Nov 17 07:29:54 2015
Name:    attraction.py
Purpose: Analyze larva motion within a Pitri dish
Author:  Andrea Vaccari (avaccari@middlebury.edu)

If you use our software, please cite our work:
@article{dombrovski_plastic_2019,
	title = {A {Plastic} {Visual} {Pathway} {Regulates} {Cooperative} {Behavior} in {Drosophila} {Larvae}},
	volume = {29},
	issn = {0960-9822},
	url = {https://www.cell.com/current-biology/abstract/S0960-9822(19)30489-0},
	doi = {10.1016/j.cub.2019.04.060},
	language = {English},
	number = {11},
	urldate = {2019-07-17},
	journal = {Current Biology},
	author = {Dombrovski, Mark and Kim, Anna and Poussard, Leanne and Vaccari, Andrea and Acton, Scott and Spillman, Emma and Condron, Barry and Yuan, Quan},
	month = jun,
	year = {2019},
	pmid = {31130457},
	pages = {1866--1876.e5},
}
@article{dombrovski_cooperative_2017,
	title = {Cooperative {Behavior} {Emerges} among {Drosophila} {Larvae}},
	volume = {27},
	issn = {0960-9822},
	url = {https://www.cell.com/current-biology/abstract/S0960-9822(17)30958-2},
	doi = {10.1016/j.cub.2017.07.054},
	language = {English},
	number = {18},
	urldate = {2019-07-17},
	journal = {Current Biology},
	author = {Dombrovski, Mark and Poussard, Leanne and Moalem, Kamilia and Kmecova, Lucia and Hogan, Nic and Schott, Elisabeth and Vaccari, Andrea and Acton, Scott and Condron, Barry},
	month = sep,
	year = {2017},
	pmid = {28918946},
	keywords = {behavior, cooperation, Drosophila, learning, vision},
	pages = {2821--2826.e2},
}


Copyright (c) 2018 Andrea Vaccari

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
# - If the larva is not detected, decide what to do with the various metrics
#   inf? NAN? 0? don't store it?
# - center of arena is the center of a square and the center of the target is
#   the center of another square
# - random location with about the same distance from the center and calculate
#   how much time is spend in something the same size
# - calculate distances and speed based on actual size of the arena (ask user
#   for diameter) *****
# - Scale back the measurements in pixels of the original image *****
# - Frequency of turns as function of distance from target (curviness of track
#   or average curvature as function of distance from target. Ratio between
#   total distance travelled and direct distance between start and end every
#   so many frames)
# - Save also as 200x200
# - Create script to average a bunch of heatmaps
# - See if you can store the number of active frames within the heatmap

import argparse
import cv2
from os.path import basename, splitext
import numpy as np
import pandas as pd
from backgroundSubtractor import createBackgroundSubtractorAVG
from skimage import feature as skif
import sys
import traceback
import time
from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.spatial import distance as dist

from userInt import userInt
from cvVideo import video

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class larva(object):
    def __init__(self):
        self.contour = []
        self.center = None
        self.totalDist = 0
        self.lastDist = 0
        self.inTarget = False

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
        self.center = np.asarray(bbox[0]) / 2.
        self.size = np.asarray(bbox[1]) / 2.
        self.norm = self.size.max()
        self.angle = np.asarray(bbox[2])
        self.bbox = (self.center, self.size, self.angle)
        self.target = None
        self.name = name
        self.rotation = None

        # Evaluate ellipse as polyline
        self.asPoly = cv2.ellipse2Poly(tuple(self.bbox[0].astype(np.int)),
                                       tuple(np.round(self.bbox[1] / 2.).astype(np.int)),
                                       self.bbox[2],
                                       0, 360, 10)

        # Get corners of rotated rectangle and sort clock-wise
        corners = cv2.boxPoints(self.bbox).astype(np.float32)
        srt = corners[np.argsort(corners[:, 0]), :]
        leftMost = srt[:2, :]
        rightMost = srt[2:, :]
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]
        self.corners = np.array([tl, tr, br, bl], dtype="float32")

        # Define perspective transformation to standard 100x100
        self.stdSize = (100, 100)  # (X, Y)
        self.stdBox = np.array([[0, 0],
                                [self.stdSize[0], 0],
                                list(self.stdSize),
                                [0, self.stdSize[1]]],
                               dtype=np.float32)
        self.perspective = cv2.getPerspectiveTransform(self.corners, self.stdBox)

        # Larvae are in regions
        self.larva = larva()  # What if more than one larva in more than one arena?

    def getScaledBBox(self, scale=1.0, fix=[0.0, 0.0]):
        return (self.center, self.size * scale + fix, self.angle)

    def getScaledPoly(self, scale=1.0, fix=[0.0, 0.0]):
        bbox = self.getScaledBBox(scale, fix)
        return cv2.ellipse2Poly(tuple(bbox[0].astype(np.int)),
                                tuple(np.round(bbox[1] / 2.).astype(np.int)),
                                bbox[2],
                                0, 360, 10)

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
        plt.ion()

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

        self.rewinded = False

        self.frameNo = 0
        self.frameDet = 0
        self.frameHistoryLen = 50
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        self.fgbg = createBackgroundSubtractorAVG(bufferSize=self.frameHistoryLen,
                                                  alpha=0.02)

        self.sourceFrame = None
        self.processedFrame = None
        self.lastProcFrame = None
        self.workingFrame = None

        self.kdeSupportStep = 0.01
        self.kdeSupport = np.arange(0, 1, self.kdeSupportStep)

        self.heatMap = None

        self.selectionWindow = 'Selection Window'
        self.arenas = []
        self.arenasNo = 0
        self.selPts = []
        self.selPtsNo = 0
        self.selectionMask = None
        self.selectionMode = False
        self.selectionType = None

        self.thresholdRadius = 0.0
        self.downscaleFrame = False
        self.frameScale = 1.0

        self.lastOutFrame = None
        self.lastOutStored = False
        self.lastOutData = None


        # Create an empty pandas dataframe to store the data
        self.columns = ['FrameNo',
                        'OrigFrameNo',
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
                        'DistTrgtCntrNorm']
        self.data = pd.DataFrame(dict.fromkeys(self.columns, []))

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
            if self.arenas[-1].larva.inTarget:  # This uses the last arena
                return True
            return False

        # If we found at least one associate it to a larva and the correct arena
        # Might have to do some filtering here (area of larva ~ 30)
        larva_center = contours[0].squeeze().mean(axis=0)

        # Find the correct arena
        for arena in self.arenas:
            arena.larva.inTarget = False
            if cv2.pointPolygonTest(arena.asPoly, tuple(larva_center), False) >= 0:
                arena.larva.clearContour()
                arena.larva.updateContour(contours[0])

                # Check if the larva is in the target
                if cv2.pointPolygonTest(arena.target.asPoly, tuple(larva_center), True) >= -self.targetPxlDist:
                    arena.larva.inTarget = True

        return True

    def preprocessFrame(self):

        # If user asked to Pyramid down
        if self.downscaleFrame:
            frame = cv2.pyrDown(self.sourceFrame, borderType=cv2.BORDER_REPLICATE)
            self.frameScale = 2.0
        else:
            frame = self.sourceFrame
            self.frameScale = 1.0

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
                    cv2.ellipse(self.selectionMask,
                                a.target.bbox,
                                [0, 0, 0],
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

            # Rewind the movie to the first frame
            if self.rewinded is False:
                self.vid.goToMs()
                self.frameNo = 0
                self.rewinded = True
                return

            # Detect larva in the foreground
            if self.detectLarva(fg) is False:
                self.processedFrame = self.sourceFrame
                return

            # Count frames where larva was detected
            self.frameDet += 1

            # Create heatmap and add to original image in HOT colormap
            if self.heatMap is None:
                self.heatMap = np.zeros_like(self.sourceFrame, dtype=np.float)

            # If larva is not in target
            if not self.arenas[-1].larva.inTarget:  # Uses last arena
                temp = np.zeros_like(self.heatMap)
                for arena in self.arenas:
                    cv2.drawContours(temp, arena.larva.contour, 0, 1, cv2.FILLED)
                self.heatMap += temp

            # Normalize and add heatmap to fame
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
                cv2.ellipse(self.processedFrame,
                            arena.bbox,
                            [255, 0, 0],
                            1)
                self.drawBox(self.processedFrame, arena.corners)
                self.drawCross(self.processedFrame,
                               arena.center.astype(np.uint16),
                               color=[255, 0, 0])

                if arena.target is not None:
                    cv2.ellipse(self.processedFrame,
                                arena.target.bbox,
                                [0, 0, 255],
                                1)
                    thickness = 1;
                    if arena.larva.inTarget:
                        thickness = -1;
                    cv2.ellipse(self.processedFrame,
                                arena.target.getScaledBBox(fix=[self.targetPxlDist, self.targetPxlDist]),
                                [128, 128, 255],
                                thickness)
                    self.drawCross(self.processedFrame,
                                   arena.target.center.astype(np.uint16))

                # Add larva contour to original image in green
                if not self.arenas[-1].larva.inTarget:  # Uses last arena
                    cv2.drawContours(self.processedFrame,
                                     arena.larva.contour,
                                     0,
                                     (0, 255, 0),
                                     1)
                    if arena.larva.center is not None:
                        self.drawCross(self.processedFrame,
                                       arena.larva.center.astype(np.uint16),
                                       color=[0, 255, 0])

                # Add larva path this far
                for coo in np.round(self.data[['LarvaPosX', 'LarvaPosY']].values).astype(np.int):
                    self.processedFrame[coo[1], coo[0], :] = (0, 255 ,0)

                # Add special ranges in dark green
                cv2.ellipse(self.processedFrame,
                            arena.getScaledBBox(scale=self.thresholdRadius),
                            [0, 127, 0],
                            1)


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
        l_tdist = larva.totalDist
        try:
            lastDeltaFrame = self.frameNo - self.data.iloc[-1]['OrigFrameNo']
        except IndexError:
            l_lvel = 0
        else:
            l_lvel = l_ldist / lastDeltaFrame
        d_ctr_n = d_ctr * norm
        d_trg_n = d_trg * norm
        l_pos_n = l_pos * norm
        l_ldist_n = l_ldist * norm
        l_lvel_n = l_lvel * norm
        l_tdist_n = l_tdist * norm

        values = [self.frameDet,
                  self.frameNo,
                  time,
                  l_pos[0],
                  l_pos[1],
                  l_ldist,
                  l_lvel,
                  l_tdist,
                  l_pos_n[0],
                  l_pos_n[1],
                  l_ldist_n,
                  l_lvel_n,
                  l_tdist_n,
                  d_ctr,
                  d_trg,
                  d_ctr_n,
                  d_trg_n]

        # Store state when entering target
        if larva.inTarget and not self.lastOutStored:
            self.lastOutData = [self.frameDet,
                                self.frameNo,
                                time,
                                l_pos[0],
                                l_pos[1],
                                l_ldist,
                                0.0,  # l_lvel
                                l_tdist,
                                0.0,  # l_pos_n[0]
                                0.0,  # l_pos_n[1]
                                0.0,  # l_ldist_n
                                0.0,  # l_lvel_n
                                0.0,  # l_tdist_n
                                d_ctr,
                                d_trg,
                                0.0,  # d_ctr_n
                                0.0]  # d_trg_n
            self.lastOutStored = True

        # Evaluate intermediate states when leaving the target
        if not larva.inTarget and self.lastOutStored:
            currentData = [self.frameDet,
                           self.frameNo,
                           time,
                           l_pos[0],
                           l_pos[1],
                           l_ldist,
                           0.0,  # l_lvel
                           l_tdist,
                           0.0,  # l_pos_n[0]
                           0.0,  # l_pos_n[1]
                           0.0,  # l_ldist_n
                           0.0,  # l_lvel_n
                           0.0,  # l_tdist_n
                           d_ctr,
                           d_trg,
                           0.0,  # d_ctr_n
                           0.0]  # d_trg_n

            # Linear interpolate missing data
            interp = np.array([np.linspace(i, j, self.frameNo - self.lastOutData[0] + 1) for i, j in zip(self.lastOutData, currentData)])

            # Calculate derived measures
            interp[5] = interp[4]  # l_lvel
            interp[[7, 8, 9, 10, 11, 14, 15]] = interp[[2, 3, 4, 5, 6, 12, 13]] * norm

            # Append data to existing dataframe
            self.data = self.data.append(pd.DataFrame(dict(zip(self.columns, interp))),
                                         ignore_index=True)
            self.lastOutStored = False
            return



        # If larva not in target, store data
        if not larva.inTarget:
            self.data = self.data.append(dict(zip(self.columns, values)),
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

    def drawBox(self, window, corners, color=[255, 0, 0], thickness=1):
        cv2.line(window, tuple(corners[0]), tuple(corners[1]), color=color, thickness=thickness)
        cv2.line(window, tuple(corners[1]), tuple(corners[2]), color=color, thickness=thickness)
        cv2.line(window, tuple(corners[2]), tuple(corners[3]), color=color, thickness=thickness)
        cv2.line(window, tuple(corners[3]), tuple(corners[0]), color=color, thickness=thickness)


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
        # Default arena
        arena = self.arenas[-1]  # Assumes a single arena

        # Extract data
        dst_arena = self.data['DistArenaCntrNorm'].values
        dst_trgt = self.data['DistTrgtCntrNorm'].values

        # Calculate statistics of distances
        dst_arena_avg = np.mean(dst_arena)
        dst_arena_med = np.median(dst_arena)
        dst_arena_stdev = np.std(dst_arena)
        dst_trgt_avg = np.mean(dst_trgt)
        dst_trgt_med = np.median(dst_trgt)
        dst_trgt_stdev = np.std(dst_trgt)

        # Calculate time/frame statistics
        frm_insd_rad = 1.0 * np.sum(dst_arena < self.thresholdRadius) / dst_arena.size

        # Create a statistics dataframe
        statData = pd.DataFrame({'DistArenaCntrAvg': dst_arena_avg,
                                 'DistArenaCntrMed': dst_arena_med,
                                 'DistArenaCntrStdev': dst_arena_stdev,
                                 'DistTrgtCntrAvg': dst_trgt_avg,
                                 'DistTrgtCntrMed': dst_trgt_med,
                                 'DistTrgtCntrStdev': dst_trgt_stdev,
                                 'FracInsdRad': frm_insd_rad},
                                 index=[0])

        # Evaluate KDE of normalized distance from center of arena
        d_arena_kde = KDEUnivariate(dst_arena)
        d_arena_kde.fit()
        d_arena_kde_density = d_arena_kde.evaluate(self.kdeSupport) * self.kdeSupportStep

        # Evaluate KDE of normalized distance from center of target
        d_trgt_kde = KDEUnivariate(dst_trgt)
        d_trgt_kde.fit()
        d_trgt_kde_density = d_trgt_kde.evaluate(self.kdeSupport) * self.kdeSupportStep

        # Plot KDEs
        plt.figure()
        plt.plot(self.kdeSupport, d_arena_kde_density, color='blue')
        plt.plot(self.kdeSupport, d_trgt_kde_density, color='red')
        plt.plot((self.thresholdRadius, self.thresholdRadius),
                 (0, plt.ylim()[1]),
                 color = 'green')
        plt.xlabel('Normalized distance')
        plt.ylabel('Probability')
        plt.title("KDE of distance from arena's (blue) and target's (red) centers")

        # Plot distance vs time and limits
        plt.figure()
        frames = self.data['FrameNo'].values
        plt.scatter(frames, dst_arena, color='blue')
        plt.scatter(frames, dst_trgt, color='red')
        plt.plot((0, frames.max()),
                 (self.thresholdRadius, self.thresholdRadius),
                 color='green')
        plt.ylim((0, 1))
        plt.xlabel('Frame number')
        plt.ylabel('Normalized distance')
        plt.title("Normalized distance from arena's (blue) and target's (red) centers")

        # Show and save original heatmap
        plt.figure()
        heatNorm = self.heatMap / self.frameDet
        plt.imshow(heatNorm)
        plt.title('Original heatmap')
        plt.imsave(splitext(self.fil)[0] + time.strftime('_HM_%Y%m%d%H%M%S') + '.png',
                   heatNorm,
                   cmap=plt.cm.hot,
                   format='png')

        # Show and save normalized, perspective-corrected, and rotated heatmap
        plt.figure()
        # Standardize the image to the template (affine transform)
        stdHeat = cv2.warpPerspective(self.heatMap, arena.perspective, arena.stdSize)
        # Evaluate the angular position of the target with respect to the center
        ctrs = np.vstack((arena.center, arena.target.center))
        perCtrs = cv2.perspectiveTransform(np.array([ctrs]), arena.perspective).squeeze()
        trgtCoo = perCtrs[1] - perCtrs[0]
        trgtCoo[1] = -trgtCoo[1]
        trgtAngle = np.arctan2(trgtCoo[1], trgtCoo[0]) - np.pi/2
        # Define the rotation required to move the target up
        rot = cv2.getRotationMatrix2D(tuple(perCtrs[0]), -np.rad2deg(trgtAngle), 1.0)
        # Apply the transformation
        stdHeat = cv2.warpAffine(stdHeat, rot, arena.stdSize)
        rotCtrs = cv2.transform(perCtrs.reshape((len(perCtrs), 1, 2)), rot).reshape((len(perCtrs), 2))
        # Normalize by the number of frames and show
        stdHeat /= self.frameDet
        plt.imshow(stdHeat)

        # Add markers for center and target center
        plt.scatter(rotCtrs[:, 0], rotCtrs[:, 1], color='red')
        plt.title('Standardized heatmap')
        plt.imsave(splitext(self.fil)[0] + time.strftime('_HMStd_%Y%m%d%H%M%S') + '.png',
                   stdHeat,
                   cmap=plt.cm.hot,
                   format='png')

        # Generate a heatmap with target and center marked and save
        mrkdHeat = stdHeat.copy()
        (y0, x0) = rotCtrs[0].astype(int)
        (y1, x1) = rotCtrs[1].astype(int)
        val = np.max(stdHeat)
        mrkdHeat[x0 - 1:x0 + 2, y0] = val
        mrkdHeat[x0, y0 - 1:y0 + 2] = val
        mrkdHeat[x1 - 1:x1 + 2, y1] = val
        mrkdHeat[x1, y1 - 1:y1 + 2] = val
        plt.imsave(splitext(self.fil)[0] + time.strftime('_HMMrkd_%Y%m%d%H%M%S') + '.png',
                   mrkdHeat,
                   cmap=plt.cm.hot,
                   format='png')

        # Evaluate 10x10 grid normalized heatmap
        coo = self.data[['LarvaPosX', 'LarvaPosY']].values
        coo = cv2.perspectiveTransform(np.array([coo]), arena.perspective).squeeze()
        coo = cv2.transform(coo.reshape((len(coo), 1, 2)), rot).reshape((len(coo), 2))
        coo = np.round(coo/10).astype(np.int)
        heat10 = np.zeros((11, 11))
        # Count frames within each box
        for c in coo:
            c = (c[1], c[0])  # (x, y) -> (r, c)
            heat10[c] += 1
        heat10 /= self.frameDet  # Normalize by the frame number
        heat10Data = pd.DataFrame(heat10)  # Create a dataframe
        plt.figure()
        plt.imshow(heat10)
        plt.title('Rasterized and normalized heatmap')

        # Show the plots
        plt.show()

        # Create KDE dataframe
        kdeData = pd.DataFrame({'DistArenaCntrNorm': self.kdeSupport,
                                'DistArenaCntrProb': d_arena_kde_density,
                                'DistTrgtCntrNorm': self.kdeSupport,
                                'DistTrgtCntrProb': d_trgt_kde_density})

        # Create assay metadata dataframe
        arena_size = arena.size * self.frameScale;
        arena_cntr = arena.center * self.frameScale;
        target_size = arena.target.size * self.frameScale;
        target_cntr = arena.target.center * self.frameScale;
        assyMetadata = pd.DataFrame({'ArenaWidth': arena_size[0],
                                     'ArenaHeight': arena_size[1],
                                     'ArenaCntrX': arena_cntr[0],
                                     'ArenaCntrY': arena_cntr[1],
                                     'TargerWidth': target_size[0],
                                     'TargetHeight': target_size[1],
                                     'TargetCntrX': target_cntr[0],
                                     'TargetCntrY': target_cntr[1],
                                     'TotalFrames': self.frameDet,
                                     'TotalOrigFrames': self.frameNo},
                                     index=[0])

        # Create and save Excel workbook
        writer = pd.ExcelWriter(splitext(self.fil)[0] + time.strftime('_%Y%m%d%H%M%S') + '.xlsx')
        self.data.to_excel(writer,
                           'Raw data',
                           columns=self.columns)

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
                                   'DistTrgtCntrStdev',
                                   'FracInsdRad'])

        assyMetadata.to_excel(writer,
                              'Metadata',
                              columns=['ArenaWidth',
                                       'ArenaHeight',
                                       'ArenaCntrX',
                                       'ArenaCntrY',
                                       'TargerWidth',
                                       'TargetHeight',
                                       'TargetCntrX',
                                       'TargetCntrY',
                                       'TotalFrames',
                                       'TotalOrigFrames'])

        heat10Data.to_excel(writer,
                            'Heatmap10',
                            header=False,
                            index=False)

        writer.save()


    def userParameters(self):
        # Should be changed to allow for user input or various parameters
        self.thresholdRadius = 0.15
        self.targetPxlDist = 5

        # Ask the user if we should downscale the frames for analysis
#        self.downscaleFrame = ui.yesNo('Downscale frame size for analysis?');
        self.downscaleFrame = True



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
                        self.userParameters()
                    self.preprocessFrame()
                    self.processFrame()
                    self.showFrame(self.mainWindow, self.processedFrame)

                # Increment total number of frames
                self.frameNo += 1

            self.processData()

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        cv2.destroyAllWindows()
        plt.close('all')


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
                exc_type, exc_value, exc_traceback = sys.exc_info()

                traceback_details = {'filename': exc_traceback.tb_frame.f_code.co_filename,
                                     'lineno'  : exc_traceback.tb_lineno,
                                     'name'    : exc_traceback.tb_frame.f_code.co_name,
                                     'type'    : exc_type.__name__,
                                     'message' : exc_value.message}

                del(exc_type, exc_value, exc_traceback)
                print traceback.format_exc()

            # Do you want to analyze another file?
            again = ui.yesNo("Do you want to open another file?")
