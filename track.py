#!/usr/local/bin/python

import cv2
import argparse
import numpy as np
import Tkinter as tk
import tkFileDialog as tkfd
import tkMessageBox as tkmb





def sorteva(eva):
    srt_idx = np.argsort(np.abs(eva))
    l1 = np.reshape(eva[srt_idx == 0], np.shape(eva)[:2])
    l2 = np.reshape(eva[srt_idx == 1], np.shape(eva)[:2])

    return np.dstack((l1, l2))


def tubularity(eva):
    seva = sorteva(eva)
    l1 = seva[:, :, 0]
    l2 = seva[:, :, 1]

    Rb = l1 / l2

    S2 = l1 * l1 + l2 * l2

    mx = 0.2 * np.max(np.sum(eva, 2))
    c = -0.5 / (mx * mx)

    v = np.exp(-2.0 * Rb * Rb) * (1.0 - np.exp(c * S2))
    v[~np.isfinite(v * l2)] = 0.0
    v[l2 >= 0] = 0.0

    return v


def hessian(array):
    (dy, dx) = np.gradient(array)
    (dydy, dxdy) = np.gradient(dy)
    (dydx, dxdx) = np.gradient(dx)
    return np.dstack((dxdx, dydx, dxdy, dydy))


def eval2ds(stack):
    a = stack[:, :, 0]
    b = stack[:, :, 1]
    c = stack[:, :, 2]
    d = stack[:, :, 3]

    T = a + d
    D = a * d - b * c

    Th = 0.5 * T
    T2 = T * T
    C = np.sqrt(T2 / 4.0 - D)

    return np.dstack((Th + C, Th - C))


def tubes(img, sigma_rng):
    img = img.astype(np.float32) / 255.0

    s = np.arange(sigma_rng[0], sigma_rng[1], 2)

    stk = np.dstack([np.empty_like(img)] * len(s))
    for i in range(len(s)):
        blr = cv2.GaussianBlur(img, (s[i], s[i]), 0)
        hes = hessian(blr)
        eva = eval2ds(hes)
        stk[:, :, i] = tubularity(eva)
    tub = 255.0 * np.max(stk, 2)

    return tub.astype(np.uint8)






class trackedArea(object):
    def __init__(self, corners):
        if corners[0][0] < corners[1][0]:
            self.c = corners[0][0]
        else:
            self.c = corners[1][0]

        if corners[0][1] < corners[1][1]:
            self.r = corners[0][1]
        else:
            self.r = corners[1][1]

        self.w = np.abs(corners[0][0] - corners[1][0])
        self.h = np.abs(corners[0][1] - corners[1][1])

        corn = []
        corn.append((self.c, self.r))
        corn.append((self.c + self.w, self.r + self.h))
        self.Corners = np.asarray(corn)

        self.templ = None
        self.templStack = None
        self.templCnt = None
        self.StackSize = None


    def initKalman(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], dtype=np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], dtype=np.float32)
        self.kf.processNoiseCov = 1.0 * np.eye(4, dtype=np.float32)
        self.kf.measurementNoiseCov = 1.0 * np.eye(2, dtype=np.float32)
        post = np.array([self.c, self.r, 0, 0], dtype=np.float32)
        post.shape = (4, 1)
        self.kf.statePost = post

    def getKalmanPredict(self):
        pred = self.kf.predict()
        pred = pred[:2].astype(np.int)
        return tuple((pred[0][0], pred[1][0]))

    def setKalmanCorrect(self, loc):
        loc = np.array(loc, dtype=np.float32)
        loc.shape = (2, 1)
        self.kf.correct(loc)

    def setStackSize(self, size):
        self.StackSize = size

    def getCorners(self):
        return [tuple(self.Corners[0]), tuple(self.Corners[1])]

    def getHalfCorners(self):
        corn = self.Corners - (self.w/2, self.h/2)
        return [tuple(corn[0]), tuple(corn[1])]

    def getEnlargedCorners(self, pxls):
        corn = self.Corners - (self.w/2, self.h/2)
        corn += np.asarray([[-pxls, -pxls], [pxls, pxls]])
        return [tuple(corn[0]), tuple(corn[1])]

    def getcrwh(self):
        return (self.c, self.r, self.w, self.h)

    def setcrwh(self, window):
        self.c = window[0]
        self.r = window[1]
        self.w = window[2]
        self.h = window[3]

    def updateWindow(self, loc):
        self.c = loc[0]
        self.r = loc[1]
        corn = []
        corn.append((self.c, self.r))
        corn.append((self.c + self.w, self.r + self.h))
        self.Corners = np.asarray(corn)

    def setTemplate(self, image):
        self.templ = image[self.r:self.r+self.h, self.c:self.c+self.w].copy()

        if self.templCnt is None:
            self.templCnt = 0
            self.templStack = np.concatenate([self.templ[..., np.newaxis] for i in range(self.StackSize)], axis=3)

        self.templCnt %= self.StackSize
        self.templStack[:, :, :, self.templCnt] = self.templ
        self.templCnt += 1

    def getGrayStackAve(self):
        ave = self.getStackAve()
        return cv2.cvtColor(ave, cv2.COLOR_BGR2GRAY)

    def getStackAve(self):
        ave = np.average(self.templStack, 3).astype(np.uint8)
        return ave

    def getGrayTemplate(self):
        return cv2.cvtColor(self.templ, cv2.COLOR_BGR2GRAY)



class watch(object):
    def __init__(self, vid):
        if vid is None:
            root = tk.Tk()
            root.withdraw()
            root.update()
            root.iconify()
            vid = tkfd.askopenfilename()

        self.showHelp('main')

        self.cap = cv2.VideoCapture(vid)

        self.sourceFrame = None
        self.processedFrame = None
        self.workingFrame = None
        self.undoFrames = []
        self.frameNo = 0
        self.lastFrame = False
        self.userInteraction = False

        self.mainWindow = 'Larvae'
        cv2.namedWindow(self.mainWindow)

        self.selectionWindow = 'Select areas to track'

        self.refPt = []
        self.tracking = False
        self.trackedAreasList = []
        self.showMatch = False

    def showHelp(self, menu):
        if menu == 'main':
            message = "Active keys:\n" + \
                      "'a' -> selects areas to track (Click, drag, release)\n" + \
                      "'m' -> toggles the display of the current template and the matching results (for trubleshooting)\n" + \
                      "'h' -> shows this help\n" + \
                      "\n'q' -> quits"
        elif menu == 'select':
            message = "Select area keys:\n" + \
                      "'l' -> clear last selection\n" + \
                      "'c' -> clear all selections\n" + \
                      "'t' -> start tracking\n" + \
                      "'h' -> shows this help\n" + \
                      "\n'q' -> quits selection"
        else:
            message = "You shouldn't be here!"

        tkmb.showinfo('Help',
                      message=message,
                      icon=tkmb.INFO)



    def mouseInteraction(self, event, x, y, flags, params):
        if self.userInteraction is True:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.refPt = [(x, y)]
            elif event == cv2.EVENT_LBUTTONUP:
                self.undoFrames.append(self.workingFrame.copy())
                self.refPt.append((x, y))
                if self.refPt[0][0] != self.refPt[1][0] and self.refPt[0][1] != self.refPt[1][1]:
                    area = trackedArea(self.refPt)
                    area.setStackSize(30)
                    area.setTemplate(self.processedFrame)
                    area.initKalman()
                    corn = area.getCorners()
                    self.trackedAreasList.append(area)

                    cv2.rectangle(self.workingFrame,
                                  corn[0], corn[1],
                                  (0, 0, 255), 1)

                    self.showFrame(self.selectionWindow, self.workingFrame)




    def selectArea(self):
        self.userInteraction = True
        cv2.namedWindow(self.selectionWindow)
        cv2.setMouseCallback(self.selectionWindow, self.mouseInteraction)
        self.workingFrame = self.processedFrame.copy()
        self.showFrame(self.selectionWindow, self.workingFrame)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.undoFrames = []
                break
            elif key == ord('c'):
                self.workingFrame = self.processedFrame.copy()
                self.trackedAreasList = []
                self.undoFrames = []
                self.showFrame(self.selectionWindow, self.workingFrame)
            elif key == ord('l'):
                try:
                    self.trackedAreasList.pop()
                except IndexError:
                    pass
                else:
                    self.workingFrame = self.undoFrames.pop()
                    self.showFrame(self.selectionWindow, self.workingFrame)
            elif key == ord('t'):
                self.undoFrames = []
                self.trackArea = self.refPt
                self.tracking = True
                break
            elif key == ord('h'):
                self.showHelp('select')



        cv2.destroyWindow(self.selectionWindow)
        self.userInteration = False




    def readFrame(self):
        ret, frame = self.cap.read()
        if ret == 0:
            self.cap.release()
            self.lastFrame = True
        else:
            self.frameNo += 1
            self.sourceFrame = frame



    def trackObjects(self):
        for area in self.trackedAreasList:
#                area = self.trackedAreasList[0]

            gray = cv2.cvtColor(self.processedFrame, cv2.COLOR_BGR2GRAY)
#            templ = area.getGrayTemplate()
            templ = area.getGrayStackAve()
            cc = cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED)
            cc = cc * cc * cc * cc
            _, cc = cv2.threshold(cc, 0.1, 0, cv2.THRESH_TOZERO)
            cc8 = cv2.normalize(cc, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            mask = np.zeros_like(cc8)
            mcorn = area.getEnlargedCorners(0)
            cv2.rectangle(mask, mcorn[0], mcorn[1], 255, -1)
            _, _, _, mx = cv2.minMaxLoc(cc8, mask)

#            kp = area.getKalmanPredict()
#            area.updateWindow(kp)
#            area.setTemplate(self.processedFrame)

            (c, r, _, _) = area.getcrwh()
            jump = 10
            if abs(c - mx[0]) < jump and abs(r - mx[1]) < jump:
#                area.setKalmanCorrect(mx)
                area.updateWindow(mx)
                area.setTemplate(self.processedFrame)

            if self.showMatch is True:
                cv2.imshow('Stack: '+str(area), templ)
                cv2.rectangle(cc8, mcorn[0], mcorn[1], 255, 1)
                cv2.circle(cc8, mx, 5, 255, 1)
                cv2.imshow('Match: '+str(area), cc8)
            else:
                try:
                    cv2.destroyWindow('Match: '+str(area))
                    cv2.destroyWindow('Stack: '+str(area))
                except:
                    pass


            corn = area.getCorners()
            cv2.rectangle(self.workingFrame,
                          corn[0], corn[1],
                          (0, 255, 0), 1)

#            self.showFrame()
#            raw_input('wait')



    def processFrame(self):
        gray = cv2.cvtColor(self.sourceFrame, cv2.COLOR_BGR2GRAY)

        tub = tubes(gray, [5, 12])
        tubular = cv2.cvtColor(tub, cv2.COLOR_GRAY2BGR)

        high = 0.3
        rest = 1.0 - high
        colorized = cv2.addWeighted(self.sourceFrame, rest, tubular, high, 0.0)
#        colorized = cv2.add(self.sourceFrame, tubular)

        self.processedFrame = np.concatenate((self.sourceFrame,
                                              tubular,
                                              colorized),
                                             axis=1)

        self.workingFrame = self.processedFrame.copy()

        if self.tracking is True:
            self.trackObjects()



    def showFrame(self, window, frame):
        cv2.imshow(window, frame)




    def watch(self):
        while self.lastFrame is False:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('m'):
                self.showMatch = not self.showMatch
            if key == ord('q'):
                break
            elif key == ord('a'):
                self.selectArea()
            elif key == ord('h'):
                self.showHelp('main')
            else:
                self.readFrame()
                self.processFrame()
                self.showFrame(self.mainWindow, self.workingFrame)

        exit()




    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file",
                        help="File to analyze.")

    args = parser.parse_args()

    watch = watch(args.file)
    watch.watch()

