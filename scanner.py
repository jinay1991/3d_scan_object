# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 All Rights Resevered
#
# Author: Jinay Patel (jinay1991@gmail.com)

import cv2
import numpy as np
import os
import logging
import random
import glob



log = logging.getLogger()
logging.basicConfig(level=logging.INFO)

class Scanner():

    def __init__(self, filename):
        """
        Contructor
        """
        random.seed(1)

        self.filename = filename
        self.in_image = cv2.imread(filename)
        if self.in_image is None:
            log.error("can not open file")
            exit(1)

        self.rows, self.cols = self.in_image.shape[:2]

        self.gray_image = None
        self.edges_image = None

        self.preprocess()

    def preprocess(self):
        """
        Perform pre-defined Pre-Processing to the Image
        """
        # generate RGB version of the input image
        self.rgb_image = cv2.cvtColor(self.in_image, cv2.COLOR_BGR2RGB)

        # noise removal
        blur = cv2.bilateralFilter(self.rgb_image, 15, 90, 90)

        # convert to GRAYSCALE
        self.gray_image = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)

    def detect_edges(self, lo_thresh=30, hi_thresh=50):
        """
        Performs Edge Detection for the Object
        """
        # DownSample
        downscaled = cv2.resize(self.gray_image, (self.cols/4, self.rows/4))

        # compute edges
        downscaled_edges = cv2.Canny(downscaled, lo_thresh, hi_thresh, apertureSize=3)

        # UpSample
        self.edges_image = cv2.resize(downscaled_edges, (self.cols, self.rows))

    def detect_contours(self, minContourArea=200):
        """
        Extracts Contour Points
        """
        if self.edges_image is None:
            self.detect_edges()

        # Find contours for the Morphed Image
        ret, t_contours, hierarchy = cv2.findContours(
            self.edges_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # sort (largest->smallest)
        t_contours = sorted(t_contours, key=cv2.contourArea, reverse=True)

        # discard contours which has area less than 20% of image area
        self.contour_pts = []
        self.contour_boxes = []
        for c in t_contours:
            if cv2.contourArea(c) < minContourArea:
                continue
            self.contour_pts.append(c)
            self.contour_boxes.append(cv2.boundingRect(c))

    def draw_results(self, dump_output=True):
        """
        Draws Result Images and displays
        """
        vis_raw = np.zeros((self.rows, self.cols, 3), dtype=np.uint8) # 3 Channel Color Image
        vis_in = self.in_image.copy()

        # draw contour properties
        for cnt in self.contour_pts:
            color = (random.randint(50,255),random.randint(150,255),random.randint(0,130))
            # draw edge/contour of object
            cv2.drawContours(vis_raw, [cnt], -1, color, thickness=2)
            cv2.drawContours(vis_in, [cnt], -1, color, thickness=4)
            # draw centroid
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(vis_in, (cx, cy), 4, color, thickness=-1)

        # disply
        if log.level == logging.DEBUG:
            plt.subplot(221), plt.title("rgb"), plt.imshow(self.rgb_image)
            plt.subplot(222), plt.title("edges"), plt.imshow(self.edges_image, cmap='gray')
            plt.subplot(223), plt.title("vis_raw"), plt.imshow(vis_raw, cmap='gray')
            plt.subplot(224), plt.title("vis_in"), plt.imshow(vis_in)
            plt.show()

        # dump result
        if dump_output:
            output_dir = os.path.join(os.path.abspath(os.path.curdir),
                                      "output",
                                      os.path.basename(os.path.dirname(self.filename)))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            dumpFileName = os.path.join(output_dir, str(os.path.splitext(os.path.basename(self.filename))[0]) + "_OUT.jpg")
            cv2.imwrite(dumpFileName, vis_in)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input file", required=True)
    parser.add_argument("--debug", help="enable debug blocks", action="store_true", default=False)
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)
        from matplotlib import pyplot as plt

    if not os.path.exists(os.path.abspath(args.input)):
        log.error("%s does not exists.", args.input)
        exit(1)

    scnr = Scanner(args.input)
    scnr.detect_edges()
    scnr.detect_contours()
    scnr.draw_results()
