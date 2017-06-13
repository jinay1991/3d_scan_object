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
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


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

        self.gray_image = None
        self.edges_image = None
        self.cropped_image = None

        self.thresh_val = 80

        self.preprocess()

    def isWithinROICircle(self, pt):
        """
        Check whether provided point lies under the ROI Circle or not.
        :returns:   True    if point is inside ROI Circle
                    False   if point is outside or on the ROI circle
        """
        (cx, cy) = pt
        circle_lhs = pow((cx - self.center_x), 2) + \
            pow((cy - self.center_y), 2)
        if circle_lhs > pow(self.radius, 2):
            return False
        else:
            return True

    def preprocess(self):
        """
        Perform pre-defined Pre-Processing to the Image
        """
        # extract dim of image
        self.rows, self.cols = self.in_image.shape[:2]

        # Valid Region for Centroid of object contour should be in circle
        self.radius = int(self.cols * 0.10)
        self.center_x = int(self.cols / 2)
        self.center_y = int(self.rows / 2)

        # generate RGB version of the input image
        self.rgb_image = cv2.cvtColor(self.in_image, cv2.COLOR_BGR2RGB)

        # noise removal
        blur = cv2.GaussianBlur(self.rgb_image, (9, 9), 0)
        blur = cv2.bilateralFilter(blur, 11, 90, 90)

        # convert to GRAYSCALE
        self.gray_image = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)

        # threshold image
        _, self.thresh_image = cv2.threshold(
            self.gray_image, self.thresh_val, 255, cv2.THRESH_BINARY_INV)

        # cropped image data [Referred as Output Result]
        self.cropped_image = self.rgb_image.copy()

    def detect_edges(self, lo_thresh=80, hi_thresh=160):
        """
        Performs Edge Detection for the Object
        """
        # scale down to avoid breakages in detected edges
        downscaled_rgb = cv2.resize(
            self.rgb_image, (self.cols / 2, self.rows / 2))
        downscaled_gray = cv2.resize(
            self.gray_image, (self.cols / 2, self.rows / 2))
        downscaled_thresh = cv2.resize(
            self.thresh_image, (self.cols / 2, self.rows / 2))

        # Canny Edge Detection
        edges_R = cv2.Canny(downscaled_rgb[:, :, 0],
                            lo_thresh, hi_thresh, apertureSize=3)
        edges_G = cv2.Canny(downscaled_rgb[:, :, 1],
                            lo_thresh, hi_thresh, apertureSize=3)
        edges_B = cv2.Canny(downscaled_rgb[:, :, 2],
                            lo_thresh, hi_thresh, apertureSize=3)
        edges_gray = cv2.Canny(downscaled_gray, lo_thresh,
                               hi_thresh, apertureSize=3)
        edges_thresh = cv2.Canny(downscaled_thresh, lo_thresh,
                                 hi_thresh, apertureSize=3)

        # Normalize the Edges
        downscaled_edges = np.max(
            np.array([edges_R, edges_B, edges_G, edges_gray, edges_thresh]), axis=0)
        mean = np.mean(downscaled_edges)
        downscaled_edges[downscaled_edges < mean] = 0

        # Connect Edge breakages
        self.edges_image = cv2.resize(downscaled_edges, (self.cols, self.rows))
        self.edges_image[self.edges_image > 0] = 255
        self.edges_image[self.thresh_image > 0] = 255

        # Morphological Closing Operation to connect the joints and fill the
        # holes/breakages
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.edges_image = cv2.morphologyEx(
            self.edges_image, cv2.MORPH_CLOSE, kernel)

    def detect_contours(self):
        """
        Extracts Contour Points and it's properties (centroid, bounding box and whether contained inside ROI)

        :returns: True  if no alignment required
                  False if alignment of object is required

        :note:    If contour count is 0, application will terminate, as for User to realign
                  the object such that it can be completely visible all the time during scanning.
        """
        if self.edges_image is None:
            self.detect_edges()

        # Find contours for the Morphed Image
        ret, t_contours, hierarchy = cv2.findContours(
            self.edges_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t_contours = [cv2.approxPolyDP(
            c, epsilon=0.004 * cv2.arcLength(c, True), closed=True) for c in t_contours]

        # discard if no continuos contour found [Request User for realign and
        # restart]
        if len(t_contours) == 0:
            log.error("object detected has fallen out of the camera capture region or is not visible. \
                        Current process is being stopped and all the data for current scan will be lost.\
                        Please re-align the object and restart the scanning")
            exit(1)

        # sort (largest->smallest)
        t_contours = sorted(t_contours, key=cv2.contourArea, reverse=True)

        # discard contours which has area less than 5% of image area
        self.contour_pts = []
        self.contour_boxes = []
        self.contour_centroids = []
        self.contour_withinROI = []
        for c in t_contours:
            if cv2.contourArea(c) < self.edges_image.size * 0.005:
                continue

            # check if contour bounding box hits the image boundaries, they likely to go off the scope.
            x, y, w, h = cv2.boundingRect(c)
            if (x + w >= self.cols) or (y + h >= self.rows) or (x <= 0) or (y <= 0):
                continue

            self.contour_boxes.append([x, y, w, h])
            self.contour_pts.append(c)

            # compute centroids and bounding box
            M = cv2.moments(c)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            self.contour_centroids.append((cx, cy))

            self.contour_withinROI.append(self.isWithinROICircle((cx, cy)))
        if len(self.contour_pts) == 0:
            log.error("object bouding box hit the boundary and it is likely that,"
                        "object may run out of the frame. Requesting user to realign the object"
                        "such a way that they do not fall out of the screen.")
            exit(1)

    def realign_object(self):
        """
        Re-align Detected Object in the Center
        """
        for i in range(len(self.contour_pts)):

            cnt = self.contour_pts[i]

            # get centroid
            (cx, cy) = self.contour_centroids[i]

            x_disp = self.center_x - cx
            y_disp = 0  # no vertical movements
            M = np.float32([[1, 0, x_disp], [0, 1, y_disp]])
            vis_in = cv2.warpAffine(
                self.cropped_image, M, (self.cols, self.rows), borderMode=cv2.BORDER_WRAP)

        # Update existing records with new
        self.in_image = cv2.cvtColor(vis_in, cv2.COLOR_RGB2BGR)
        self.preprocess()
        self.detect_edges()
        self.detect_contours()

    def dump_results(self, frame, idx=0, directory="output"):
        output_dir = os.path.join(os.path.abspath(os.path.curdir),
                                      directory,
                                      os.path.basename(os.path.dirname(self.filename)))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        dumpFileName = os.path.join(output_dir, "%d_OUT.jpg" % int(idx))
        dump_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dumpFileName, dump_img)
        if log.level == logging.DEBUG:
            plt.imshow(frame), plt.show()


    def draw_results(self, dump_output=True):
        """
        Draws Result Images and displays
        """
        vis_raw = np.zeros((self.cols, self.rows, 3),
                           dtype=np.uint8)  # 3 Channel Color Image
        vis_in = self.rgb_image.copy()

        # draw contour properties
        for i in range(len(self.contour_pts)):
            log.debug("processing contour[%s]", i)
            cnt = self.contour_pts[i]
            color = (random.randint(150, 255), random.randint(
                0, 255), random.randint(0, 255))

            # draw edge/contour of object
            cv2.drawContours(vis_raw, [cnt], -1, color, thickness=2)
            cv2.drawContours(vis_in, [cnt], -1, color, thickness=2)

            # draw bounding box
            x, y, w, h = self.contour_boxes[i]
            if (x + w >= self.cols) or (y + h >= self.rows) or (x <= 0) or (y <= 0):
                cv2.rectangle(vis_in, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
            else:
                cv2.rectangle(vis_in, (x, y), (x + w, y + h), color, thickness=2)

            # draw centroid
            (cx, cy) = self.contour_centroids[i]
            cv2.circle(vis_in, (cx, cy), 4, color, thickness=-1)

        # draw center point
        cv2.circle(vis_in, (self.center_x, self.center_y),
                   2, (255, 0, 0), thickness=-1)

        # display
        if log.level == logging.DEBUG:
            plt.subplot(221), plt.title("rgb"), plt.imshow(self.rgb_image)
            plt.subplot(222), plt.title("edges"), plt.imshow(
                self.edges_image, cmap='gray')
            plt.subplot(223), plt.title(
                "cropped"), plt.imshow(self.cropped_image)
            plt.subplot(224), plt.title("vis_in"), plt.imshow(vis_in)
            plt.show()

        # dump result
        if dump_output:
            self.dump_results(vis_in,
                              idx=str(os.path.splitext(os.path.basename(self.filename))[0]),
                              directory="output")


class MultiScan():
    """
    Class to process over multiple scan images
    """

    def scan_single(self, image):
        """
        Perform Single Scan
        """
        scnr = Scanner(image)
        scnr.detect_edges()
        scnr.detect_contours()
        scnr.realign_object()
        scnr.draw_results()

        return scnr

    def matchPoints(self, image_1, image_2):
        # initiate ORB detector
        self.orb = cv2.ORB_create()

        # find keypoints and descriptors for both images with ORB
        kp1, des1 = self.orb.detectAndCompute(image_1, None)
        kp2, des2 = self.orb.detectAndCompute(image_2, None)

        # draw only keypoints location, not size and orientation
        kp1_image = cv2.drawKeypoints(
            image_1, kp1, None, color=(0, 255, 0), flags=0)
        kp2_image = cv2.drawKeypoints(
            kp1_image, kp2, None, color=(0, 0, 255), flags=0)

        # create BFMatcher object
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # match descriptors
        self.matches = self.bf.match(des1, des2)

        # Sort them in the order of their distance.
        self.matches = sorted(self.matches, key=lambda x: x.distance)

        # draw first 10 matches
        matches_image = np.zeros(
            (image_1.shape[0] * 2, image_1.shape[1] * 2), dtype=np.uint8)
        matches_image = cv2.drawMatches(image_1, kp1, image_2, kp2, self.matches[
                                        :20], matches_image, flags=2)

        if log.level == logging.DEBUG:
            from matplotlib import pyplot as plt
            plt.imshow(matches_image), plt.title("matches_image")
            # plt.subplot(122), plt.imshow(kp2_image), plt.title("kp2_image")
            plt.show()

    

    def find_crop_dimension(self):
        max_obj_width = 0
        max_obj_height = 0
        for scan in self.scanlist:
            box_area = np.asarray([ w * h for x, y, w, h in scan.contour_boxes])
            max_index = box_area.argmax()
            max_box = scan.contour_boxes[max_index]

            max_obj_width = max(max_obj_width, max_box[2])
            max_obj_height = max(max_obj_height, max_box[3])
        
        log.info("max crop dimensions: %sx%s for frame dimensions: %sx%s", 
                    max_obj_width, max_obj_height,
                    scan.cols, scan.rows)
        return (max_obj_width, max_obj_height)

    def crop_frame(self):
        i = 0
        for scan in self.scanlist:

            rows, cols = scan.rgb_image.shape[:2]
            (cx, cy) = (scan.center_x, scan.center_y)
            (bx, by) = (int(self.obj_width / 2), int(self.obj_height / 2))
            (dx, dy) = (cx - bx, cy - by)
            x = cx - bx
            y = cy - by
            w = self.obj_width
            h = self.obj_height

            # draw contour
            vis_in = scan.rgb_image.copy()
            cv2.drawContours(vis_in, scan.contour_pts, -1, (255, 0, 0), thickness=2)


            frame = vis_in[:, x : x + w].copy()
            scan.dump_results(frame, idx=i, directory="frames")
            i += 1

    def scan_all(self, images):
        """
        Perform multi scan
        """
        if ".jpg" in os.path.basename(images):
            filelist = [images]
        else:
            log.setLevel(logging.INFO)
            filelist = glob.glob(images + "/*.jpg")

            # sort filelist
            utils.sort_list(filelist)

        # scan frames
        self.scanlist = []
        for filepath in filelist:
            log.info("Processing %s", filepath)
            self.scanlist.append(self.scan_single(filepath))

        log.info("Total number of Scans processed %d", len(self.scanlist))

        self.obj_width, self.obj_height = self.find_crop_dimension()
        self.crop_frame()

        if not (".jpg" in os.path.basename(images)):
            utils.convert2video(os.path.join(os.path.abspath(os.path.curdir), "output", os.path.basename(args.input)),
                                video_name=os.path.basename(
                                    args.input) + "_OUT.avi",
                                fps=5.0)
            log.info("now saving cropped version")
            utils.convert2video(os.path.join(os.path.abspath(os.path.curdir), "frames", os.path.basename(args.input)), 
                                video_name=os.path.basename(
                                    args.input) + "_framesOUT.avi",
                                fps=5.0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input file", required=True)
    parser.add_argument("--debug", help="enable debug blocks",
                        action="store_true", default=False)
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)
        from matplotlib import pyplot as plt

    if not os.path.exists(os.path.abspath(args.input)):
        log.error("%s does not exists.", args.input)
        exit(1)

    scanner = MultiScan()
    scanner.scan_all(args.input)
