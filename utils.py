# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 All Rights Resevered
#
# Author: Jinay Patel (jinay1991@gmail.com)

import cv2
import numpy as np
import os
import logging
import glob

log = logging.getLogger()
logging.basicConfig(level=logging.INFO)

def convert2video(image_seq_dir, video_name='video.avi'):
    """
    Converts provided Image sequece present in dir to Video.
    """
    if os.path.exists(os.path.join(os.path.abspath(os.path.curdir), video_name)):
        os.remove(os.path.join(os.path.abspath(os.path.curdir), video_name))
    first_image = cv2.imread(os.path.join(image_seq_dir, "0.jpg"))
    height , width , layers =  first_image.shape

    frameId = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name,fourcc,10.0,(width,height), isColor=True)
    filelist = glob.glob(os.path.join(image_seq_dir, "*.jpg"))
    log.info("len(filelist): %s", len(filelist))
    for frameId in range(len(filelist)):
        filepath = os.path.join(image_seq_dir, "%d.jpg" % frameId)
        frame = cv2.imread(filepath)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="dir path containing image sequence (0.jpg, 1.jpg, ...)", required=True)
    parser.add_argument("--debug", help="enable debug blocks", action="store_true", default=False)
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    if not os.path.exists(os.path.abspath(args.dir)):
        log.error("%s does not exists.", args.dir)
        exit(1)

    convert2video(args.dir)