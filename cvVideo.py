"""
Created on Sat Jan 23 12:07:33 2016
Name:    cvVideo.py
Purpose: video driver based on opencv
Author:  Andrea Vaccari (avaccari@middlebury.edu)

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

import cv2
from os.path import basename


class video(object):
    def __init__(self, *args, **kwargs):
        self.nextFrameNo = None
        self.totalFrames = None
        self.cap = None

        self.properties = {
            cv2.CAP_PROP_POS_MSEC : ('CV_CAP_PROP_POS_MSEC',
                                     'Current position of the video file in milliseconds.'),
            cv2.CAP_PROP_POS_FRAMES : ('CV_CAP_PROP_POS_FRAMES',
                                       '0-based index of the frame to be decoded/captured next.'),
            cv2.CAP_PROP_POS_AVI_RATIO : ('CV_CAP_PROP_POS_AVI_RATIO',
                                          'Relative position of the video file: 0 - start of the film, 1 - end of the film.'),
            cv2.CAP_PROP_FRAME_WIDTH : ('CV_CAP_PROP_FRAME_WIDTH',
                                        'Width of the frames in the video stream.'),
            cv2.CAP_PROP_FRAME_HEIGHT : ('CV_CAP_PROP_FRAME_HEIGHT',
                                         'Height of the frames in the video stream.'),
            cv2.CAP_PROP_FPS : ('CV_CAP_PROP_FPS',
                                'Frame rate.'),
            cv2.CAP_PROP_FOURCC : ('CV_CAP_PROP_FOURCC',
                                   '4-character code of codec.'),
            cv2.CAP_PROP_FRAME_COUNT : ('CV_CAP_PROP_FRAME_COUNT',
                                        'Number of frames in the video file.'),
            cv2.CAP_PROP_FORMAT : ('CV_CAP_PROP_FORMAT',
                                   'Format of the Mat objects returned by retrieve().'),
            cv2.CAP_PROP_MODE : ('CV_CAP_PROP_MODE',
                                 'Backend-specific value indicating the current capture mode.'),
            cv2.CAP_PROP_BRIGHTNESS : ('CV_CAP_PROP_BRIGHTNESS',
                                       'Brightness of the image (only for cameras).'),
            cv2.CAP_PROP_CONTRAST : ('CV_CAP_PROP_CONTRAST',
                                     'Contrast of the image (only for cameras).'),
            cv2.CAP_PROP_SATURATION : ('CV_CAP_PROP_SATURATION',
                                       'Saturation of the image (only for cameras).'),
            cv2.CAP_PROP_HUE : ('CV_CAP_PROP_HUE',
                                'Hue of the image (only for cameras).'),
            cv2.CAP_PROP_GAIN : ('CV_CAP_PROP_GAIN',
                                 'Gain of the image (only for cameras).'),
            cv2.CAP_PROP_EXPOSURE : ('CV_CAP_PROP_EXPOSURE',
                                     'Exposure (only for cameras).'),
            cv2.CAP_PROP_CONVERT_RGB : ('CV_CAP_PROP_CONVERT_RGB',
                                        'Boolean flags indicating whether images should be converted to RGB.'),
            cv2.CAP_PROP_WHITE_BALANCE_BLUE_U : ('CV_CAP_PROP_WHITE_BALANCE_BLUE_U',
                                            'The U value of the whitebalance setting (note: only supported by DC1394 v 2.x backend currently).'),
            cv2.CAP_PROP_WHITE_BALANCE_RED_V : ('CV_CAP_PROP_WHITE_BALANCE_RED_V',
                                            'The V value of the whitebalance setting (note: only supported by DC1394 v 2.x backend currently).'),
            cv2.CAP_PROP_RECTIFICATION : ('CV_CAP_PROP_RECTIFICATION',
                                          'Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently).'),
            cv2.CAP_PROP_ISO_SPEED : ('CV_CAP_PROP_ISO_SPEED',
                                      'The ISO speed of the camera (note: only supported by DC1394 v 2.x backend currently).'),
            cv2.CAP_PROP_BUFFERSIZE : ('CV_CAP_PROP_BUFFERSIZE',
                                       'Amount of frames stored in internal buffer memory (note: only supported by DC1394 v 2.x backend currently).')}

        N = len(args)
        if N == 0:
            pass
        elif N == 1:
            self.vid = args[0]
            self.open(self.vid)
        else:
            raise ValueError('Invalid number of parameters for video constructor')


    def open(self, video):
        if self.cap is not None:
            self.cap.close()
        self.cap = cv2.VideoCapture(video)

        if self.cap.isOpened():
            self.totalFrames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.nextFrameNo = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            raise IOError('Could not open file "' + self.vid)


    def readFrame(self, decode=True):
        ret = self.cap.grab()

        if ret is not True:
            return None

        if decode is False:
            return None

        ret, frm = self.cap.retrieve()

        if ret is not True:
            return None

        self.nextFrameNo = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

        return frm


    def getFrameTemplate(self):
        # Read the next frame then rewind
        location = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        frm = self.readFrame()
        self.goToMs(location)

        # If we read a frame, return it
        if frm is None:
            return None

        return frm


    def getNextFrameNo(self):
        return self.nextFrameNo


    def isFrameAvailable(self):
        return self.nextFrameNo < self.totalFrames


    def getFrameRate(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def getMs(self):
        return self.cap.get(cv2.CAP_PROP_POS_MSEC)


    # We need to move around using ms because of the with some video encoding
    # algorithms, using CAP_PROP_POS_FRAMES will jump to a 'key' frame instead
    # of an actual frame. The readback seems to be working fine.
    def goToMs(self, location=0.0):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, location)
        self.nextFrameNo = self.cap.get(cv2.CAP_PROP_POS_FRAMES)


    def play(self, start=None, stop=None, step=None):
        if start is not None:
            self.goToMs(start)
        else:
            self.goToMs()

        while self.isFrameAvailable():
            if stop is not None and self.nextFrameNo >= stop:
                break

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            frm = self.readFrame()

            if frm is not None:
                print self.nextFrameNo
                cv2.imshow(basename(self.vid), frm)

            if step is not None:
                self.goToMs(self.getMs() + step)


        cv2.destroyWindow(basename(self.vid))


    def getInfo(self, props=None):
        if props is None:
            props = self.properties

        for (k, d) in props.iteritems():
            p = self.cap.get(k)
            print '{0} -> {1}'.format(d[0], p)


    def close(self):
        self.cap.release()


    def __enter__(self):
        return self


    def __exit__(self, exec_type, exec_value, traceback):
        self.cap.release()
