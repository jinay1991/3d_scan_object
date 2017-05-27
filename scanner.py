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

import utils


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
        blur = cv2.GaussianBlur(self.rgb_image, (9, 9), 0)
        blur = cv2.bilateralFilter(blur, 15, 90, 90)

        # convert to GRAYSCALE
        self.gray_image = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)

    def detect_edges(self, lo_thresh=80, hi_thresh=160):
        """
        Performs Edge Detection for the Object
        """
        _, self.thresh_image = cv2.threshold(self.gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        downscaled_rgb = cv2.resize(self.rgb_image, (self.cols/2, self.rows/2))
        # ---------------------
        # Canny Edge Detection
        # ---------------------
        edges_R = cv2.Canny(downscaled_rgb[:,:,0], lo_thresh, hi_thresh, apertureSize=3)
        edges_G = cv2.Canny(downscaled_rgb[:,:,1], lo_thresh, hi_thresh, apertureSize=3)
        edges_B = cv2.Canny(downscaled_rgb[:,:,2], lo_thresh, hi_thresh, apertureSize=3)
        downscaled_edges = np.max(np.array([edges_R, edges_B, edges_G]), axis=0)
        mean = np.mean(downscaled_edges)
        downscaled_edges[downscaled_edges < mean] = 0

        self.edges_image = cv2.resize(downscaled_edges, (self.cols, self.rows))

    def detect_contours(self):
        """
        Extracts Contour Points
        """
        if self.edges_image is None:
            self.detect_edges()

        # Find contours for the Morphed Image
        ret, t_contours, hierarchy = cv2.findContours(
            self.edges_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t_contours = [cv2.approxPolyDP(
            c, epsilon=0.002 * cv2.arcLength(c, True), closed=True) for c in t_contours]

        # sort (largest->smallest)
        t_contours = sorted(t_contours, key=cv2.contourArea, reverse=True)

        # discard contours which has area less than 5% of image area
        self.contour_pts = []
        self.contour_boxes = []
        for c in t_contours:
            if cv2.contourArea(c) < self.edges_image.size * 0.01:
                continue
            self.contour_pts.append(c)
            self.contour_boxes.append(cv2.boundingRect(c))

    def draw_results(self, dump_output=True):
        """
        Draws Result Images and displays
        """
        vis_raw = np.zeros((self.rows, self.cols, 3), dtype=np.uint8) # 3 Channel Color Image
        vis_in = self.rgb_image.copy()

        # draw contour properties
        for i in range(len(self.contour_pts)):
            log.debug("processing contour[%s]", i)
            cnt = self.contour_pts[i]
            color = (random.randint(150,255),random.randint(0,255),random.randint(0,255))
            # draw edge/contour of object
            cv2.drawContours(vis_raw, [cnt], -1, color, thickness=2)
            cv2.drawContours(vis_in, [cnt], -1, color, thickness=2)
            # x,y,w,h = self.contour_boxes[i]
            # cv2.rectangle(vis_in,(x,y),(x+w,y+h),color,thickness=2)
            # draw centroid
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(vis_in, (cx, cy), 4, color, thickness=-1)

        # disply
        if log.level == logging.DEBUG:
            plt.subplot(221), plt.title("rgb"), plt.imshow(self.rgb_image)
            plt.subplot(222), plt.title("edges"), plt.imshow(self.edges_image, cmap='gray')
            plt.subplot(223), plt.title("thresh"), plt.imshow(self.thresh_image, cmap='gray')
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
            dump_img = cv2.cvtColor(vis_in, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dumpFileName, dump_img)

def run_single(image):
    scnr = Scanner(image)
    scnr.detect_edges()
    scnr.detect_contours()
    scnr.draw_results()
    
def run_all(image_dir):
    filelist = glob.glob(image_dir + "/*.jpg")
    for i in range(len(filelist)):
        scnr = Scanner(filelist[i])
        scnr.detect_edges()
        scnr.detect_contours()
        scnr.draw_results()
    utils.convert2video(os.path.join(os.path.abspath(os.path.curdir), "output", os.path.basename(args.input)), video_name=os.path.basename(args.input) + "_OUT.avi")


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

    # run_single(args.input)
    run_all(args.input)