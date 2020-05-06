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
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from sklearn.metrics.pairwise import cosine_similarity

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
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-m2", "--model2", required=True, type=str,
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
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.3,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")

    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def pre_process(frame, net_input_shape):
    p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose(2, 0, 1)
    # p_frame = np.expand_dims(p_frame, axis=1)
    p_frame = p_frame.reshape(1, *p_frame.shape)
    return p_frame


def imshow(name, frame):
    cv2.imshow('output', imutils.resize(frame, width=900))

def reidentification(networkReIdentification, crop_person, identification_input_shape, total_unique_persons, conf):
    idetification_frame = pre_process(crop_person, net_input_shape=identification_input_shape)
    networkReIdentification.exec_net(idetification_frame)
    if networkReIdentification.wait() == 0:  # 256 dimentional unique descriptor
        ident_output = networkReIdentification.get_output()
        for i in range(len(ident_output)):
            if (len(total_unique_persons) == 0):
                # print(ident_output[i].reshape(1,-1).shape)
                total_unique_persons.append(ident_output[i].reshape(1, -1))
            else:
                # print("Checking SIMILARITY WITH PREVIOUS PEOPLE IF THEY MATCH THEN ALTERTING PERSON COMES SECONF TIME ELSE INCREMENTING TOTAL PEOPLE")
                newFound = True
                detected_person = ident_output[i].reshape(1, -1)
                for index in range(len(total_unique_persons)):  # checking that detected person is in out list or not
                    similarity = cosine_similarity(detected_person, total_unique_persons[index])[0][0]
                    # print(similarity)
                    if similarity > 0.65: #0.58
                        # print("SAME PERSON FOUD")
                        # print(str(similarity) + "at "+str(index))
                        newFound = False
                        total_unique_persons[index] = detected_person  # updating detetected one
                        break

                if newFound and conf > 0.90:
                    total_unique_persons.append(detected_person)
                    # print('NEW PERSON FOUND')
        # print(len(total_unique_persons))
        return total_unique_persons


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
        prob_threshold = 0.3

    ### TODO: Load the model through `infer_network` ###
    network.load_model(args.model, args.cpu_extension, args.device)
    pedestrian_input_shape = network.get_input_shape()

    networkReIdentification = Network()
    networkReIdentification.load_model(args.model2, args.cpu_extension, args.device)
    identification_input_shape = networkReIdentification.get_input_shape()
    # print('Models Loaded Successfully')

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    fps = FPS().start()
    ### TODO: Loop until stream is over ###

    last_detection_time = None
    start = None

    total_unique_persons = []
    while (cap.isOpened()):
        ### TODO: Read from the video capture ###
        isAnyFrameLeft, frame = cap.read()
        width = int(cap.get(3))
        height = int(cap.get(4))
        ### TODO: Pre-process the image as needed ###
        if not isAnyFrameLeft:
            sys.stdout.flush()
            break
        displayFrame = frame.copy()

        processed_frame = pre_process(frame, net_input_shape=pedestrian_input_shape)
        ### TODO: Start asynchronous inference for specified request ###
        inference_start_time = time.time()
        network.exec_net(processed_frame)
        ### TODO: Wait for the result ###
        last_x_min = 0
        last_x_max = 0
        last_y_max = 0
        last_y_min = 0

        if network.wait() == 0:
            inference_end_time = time.time()
            total_inference_time = inference_end_time - inference_start_time
            cv2.putText(displayFrame, "Inference time: " + str(round(total_inference_time * 1000, 3)) + "ms", (5, 15),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, (230, 50, 2),
                        lineType=cv2.LINE_8, thickness=1)

            # print("Inference Time "+ total_inference_time)
            ### TODO: Get the results of the inference request ###
            result = network.get_all_output()

            ### TODO: Extract any desired stats from the results ###
            output = result['DetectionOutput']
            counter = 0

            for detection in output[0][0]:
                image_id, label, conf, x_min, y_min, x_max, y_max = detection

                if conf > prob_threshold:
                    # print("label " + str(label) + "imageid"+ str(image_id))
                    x_min = int(x_min * width)
                    x_max = int(x_max * width)
                    y_min = int(y_min * height)
                    y_max = int(y_max * height)

                    try:
                        if conf > 0.85:
                            crop_person = frame[y_min:y_max, x_min:x_max]
                            total_unique_persons = reidentification(networkReIdentification, crop_person,
                                                                    identification_input_shape, total_unique_persons, conf)

                    except Exception as err:
                        # print(err)
                        pass
                    # print(err)

                    x_min_diff = last_x_min - x_min
                    x_max_diff = last_x_max - x_max

                    if x_min_diff > 0 and x_max_diff > 0:  # ignore multiple drawn bounding boxes
                        # cv2.waitKey(0)
                        continue

                    y_min_diff = abs(last_y_min) - abs(y_min)

                    counter = counter + 1

                    last_x_min = x_min
                    last_x_max = x_max
                    last_y_max = y_max
                    last_y_min = y_min

                    cv2.rectangle(displayFrame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    activity = ""
                    # print("Y  => " + str(y_min_diff) + " " + str(y_max_diff))
                    if (y_min_diff >= -20):
                        activity = "standing"
                    elif y_min_diff < -21 and y_min_diff > -41:
                        activity = "moving"
                    else:
                        activity = "walking"

                    cv2.putText(displayFrame, activity, (x_max + 10, y_min + 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (230, 50, 2),
                                lineType=cv2.LINE_8, thickness=1)

                    last_detection_time = datetime.now()
                    # print(total_detected)
                    if start is None:
                        start = time.time()
                        time.clock()


                # cv2.putText(displayFrame, "Totol Unique Persons: "+str(len(total_unique_persons)),(50,150),
                #             cv2.FONT_HERSHEY_COMPLEX, 1, (100, 150, 250),
                #             lineType=cv2.LINE_4, thickness=2)

                # if start is not None and counter == 0:
                #     elapsed = time.time() - start
                #     client.publish("person/duration", json.dumps({"duration": elapsed}))
                #     start = None

                if last_detection_time is not None:
                    # if last_detection_time.minute
                    second_diff = (datetime.now() - last_detection_time).total_seconds()
                    # print(second_diff)
                    if second_diff >= 1.5:
                        if start is not None:
                            elapsed = time.time() - start
                            client.publish("person/duration", json.dumps({"duration": elapsed - second_diff}))
                            # start = None
                            last_detection_time = None
                            start = None


            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            client.publish("person", json.dumps({"count": str(counter), "total": len(total_unique_persons)}))
            ### Topic "person/duration": key of "duration" ###


        sys.stdout.buffer.write(displayFrame)
        #
        # imshow("frame", displayFrame)

        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###

        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.stdout.flush()
            break


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    # args = build_argparser().parse_args(args=['-i', 'resources/Pedestrian_Detect_2_1_1.mp4',
    #                                           '-m',
    #                                           'models/tensorflow/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml',
    #                                           '-m2',
    #                                           'models/intel/person-reidentification-retail-0248/FP16/person-reidentification-retail-0248.xml',
    #                                           '-d', 'CPU'])

    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
