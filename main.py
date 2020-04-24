"""People Counter."""
import imutils

"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import socket
import json
import cv2
from imutils.video import FPS
from datetime import datetime
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=False, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=False, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")

    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = None

    return client


def pre_process(frame, net_input_shape):
    p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose(2, 0, 1)
    # p_frame = np.expand_dims(p_frame, axis=1)
    p_frame = p_frame.reshape(1, *p_frame.shape)
    return p_frame


def imshow(name, frame):
    cv2.imshow('output', imutils.resize(frame, width=900))


def filter_out_same_person_detecions(output):
    print(output.shape)


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    network = Network()
    # Set Probability threshold for detections
    if not args.prob_threshold is None:
        prob_threshold = args.prob_threshold
    else:
        prob_threshold = 0.4

    ### TODO: Load the model through `infer_network` ###
    network.load_model(args.model, args.cpu_extension, args.device)
    pedestrian_input_shape = network.get_input_shape()
    print("Model Loaded Successfully ")

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    fps = FPS().start()
    ### TODO: Loop until stream is over ###

    total_people = 0
    last_detection_time = None
    while (cap.isOpened()):
        ### TODO: Read from the video capture ###
        isAnyFrameLeft, frame = cap.read()
        width = int(cap.get(3))
        height = int(cap.get(4))
        ### TODO: Pre-process the image as needed ###
        if not isAnyFrameLeft:
            break
        processed_frame = pre_process(frame, net_input_shape=pedestrian_input_shape)
        ### TODO: Start asynchronous inference for specified request ###
        network.exec_net(processed_frame)
        ### TODO: Wait for the result ###
        last_x_min = 0
        last_x_max = 0
        last_y_max = 0
        last_y_min = 0
        if network.wait() == 0:
            ### TODO: Get the results of the inference request ###
            result = network.get_all_output()

            ### TODO: Extract any desired stats from the results ###
            output = result['detection_out']
            counter = 0
            for detection in output[0][0]:
                image_id, label, conf, x_min, y_min, x_max, y_max = detection

                # print(str(label))
                if conf > 0.7:
                    x_min = int(x_min * width)
                    x_max = int(x_max * width)
                    y_min = int(y_min * height)
                    y_max = int(y_max * height)

                    x_min_diff = last_x_min - x_min
                    x_max_diff = last_x_max - x_max

                    if x_min_diff  > 0 and x_max_diff > 0: #ignore multiple drawn bounding boxes
                        # cv2.waitKey(0)
                        continue

                    y_min_diff = abs(last_y_min) - abs(y_min)
                    y_max_diff = abs(last_y_max) - abs(y_max)


                    counter = counter + 1
                    # print("X  => " + str(x_min_diff) + " " + str(x_max_diff) + " label" + str(label))
                    # print(" label" + str(label))
                    # print("Y  => " + str(y_min_diff) + " " + str(y_max_diff))

                    # print(str(y_min_diff)+ " " + str(y_max_diff))
                    last_x_min = x_min
                    last_x_max = x_max
                    last_y_max = y_max
                    last_y_min = y_min

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    activity = ""
                    if (abs(y_min_diff) <= 20):
                        activity = "stayed"
                    else:
                        activity = "Walking"

                    cv2.putText(frame, activity, (x_min, y_min), cv2.FONT_HERSHEY_PLAIN, 1, (255, 50, 50),
                                lineType=cv2.LINE_4, thickness=2)

                    last_detection_time = datetime.now()
                    # print(total_detected)


                totalPerson = "Crowd Count: " + str(counter)
                cv2.putText(frame, totalPerson, (int(width / 4), 100), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 20, 80),
                            lineType=cv2.LINE_8, thickness=2)

                cv2.putText(frame, "Totol : "+str(total_people),(100, 100), cv2.FONT_HERSHEY_PLAIN, 1, (250, 50, 250),
                            lineType=cv2.LINE_4, thickness=2)

                if last_detection_time is not None:
                    # if last_detection_time.minute
                    second_diff = (datetime.now() - last_detection_time).total_seconds()
                    print(second_diff)
                    if second_diff >= 1:
                        print(second_diff)
                        total_people += counter
                        last_detection_time = None


        # # print(total_detected)
        #     if last_detection_time is not None:
        #         # if last_detection_time.minute
        #         second_diff = (datetime.now() - last_detection_time).total_seconds()
        #         if second_diff >= 1:
        #             # print(second_diff)
        #             total_people += totalPerson
        #             last_detection_time = None




        imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    # python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/intel/pedestrian-detection-adas-0002/FP16/pedestrian-detection-adas-0002.xml -d CPU
    main()
