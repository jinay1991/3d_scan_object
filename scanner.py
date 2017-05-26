# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 All Rights Resevered
#
# Author: Jinay Patel (jinay1991@gmail.com)

import cv2
import numpy as np
import os
import logging


log = logging.getLogger()
logging.basicConfig(level=logging.INFO)

class Scanner():

    def __init__(self, filename):
        """
        Contructor
        """
        self.filename = filename
        self.in_image = cv2.imread(filename)
        if self.in_image is None:
            log.error("can not open file")
            exit(1)

        self.rows, self.cols = self.in_image.shape[:2]

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

    def __watershed_segmentation(self, image):
        ret, markers = cv2.connectedComponents(image)
        markers = markers + 1
        temp = self.in_image.copy()
        markers = cv2.watershed(temp, markers)
        temp[markers == -1] = [0, 0, 0]
        temp[markers == 1] = [255, 255, 255]
        log.info("markers: {}".format(markers))

        if log.level == logging.DEBUG:
            plt.subplot(121), plt.imshow(markers)
            plt.subplot(122), plt.imshow(temp)
            plt.show()

        return markers

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

        if log.level == logging.DEBUG:
            plt.subplot(131), plt.title("rgb"), plt.imshow(self.rgb_image)
            plt.subplot(132), plt.title("gray"), plt.imshow(self.gray_image, cmap='gray')
            plt.subplot(133), plt.title("edges"), plt.imshow(self.edges_image, cmap='gray')
            plt.show()

    def __compute_contours(self, image,minContourArea=100):
        """
        Computes Contours for the provided Image. [Private Method]
        """
        # Find contours for the Morphed Image
        ret, t_contours, hierarchy = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t_contours = [cv2.approxPolyDP(
            c, epsilon=2.5, closed=True) for c in t_contours]

        # pick largest contours
        t_contours = sorted(t_contours, key=cv2.contourArea, reverse=True)

        # discard contours which has area less than 20% of image area
        contours = []
        for c in t_contours:
            if cv2.contourArea(c) < minContourArea:
                continue
            contours.append(c)

        return contours

    def detect_contours(self):
        """
        Extracts Contour Points
        """
        self.contour_pts = self.__compute_contours(self.edges_image)

        if log.level == logging.DEBUG:
            vis_raw = np.zeros((self.rows, self.cols, 3), dtype=np.uint8) # 3 Channel Color Image
            vis_in = self.in_image.copy()
            for cnt in self.contour_pts:
                color = (random.randint(50,255),random.randint(150,255),random.randint(0,130))
                cv2.drawContours(vis_raw, [cnt], -1, color, thickness=2)
                cv2.drawContours(vis_in, [cnt], -1, color, thickness=4)

            # dump result
            dumpFileName = str(os.path.splitext(os.path.basename(self.filename))[0]) + "_OUT.jpg"
            cv2.imwrite(dumpFileName, vis_in)

            plt.subplot(131), plt.title("rgb"), plt.imshow(self.rgb_image)
            plt.subplot(132), plt.title("vis_raw"), plt.imshow(vis_raw, cmap='gray')
            plt.subplot(133), plt.title("vis_in"), plt.imshow(vis_in)
            plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input file", required=True)
    parser.add_argument("--debug", help="enable debug blocks", action="store_true", default=False)
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)
        import random
        from matplotlib import pyplot as plt
        random.seed(1)
    
    if not os.path.exists(os.path.abspath(args.input)):
        log.error("%s does not exists.", args.input)
        exit(1)
    
    scnr = Scanner(args.input)
    scnr.preprocess()
    scnr.detect_edges()
    scnr.detect_contours()
    