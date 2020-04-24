#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.core = None
        self.network = None
        self.input = None
        self.output = None
        self.exec_network = None

    def load_model(self, model_xml, extension, device):
        '''
        @model_xml XML file of IR
        @extension EXTENSION linker library *.so supported by system such as Linux, Max etc
        @device Platform CPU/GPU
        '''
        ### TODO: Load the model ###
        # self.network = IENetwork(model=model_xml, weights=os.path.splitext(model_xml)[0] + ".bin") #removed because got deprecated warning
        self.core = IECore()
        self.network = self.core.read_network(model=str(model_xml),
                                              weights=str(os.path.splitext(model_xml)[0] + ".bin"))

        ### TODO: Check for supported layers ###
        supported_layers = self.core.query_network(network=self.network, device_name=device)
        unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            print("Please check extention for these unsupported layers =>" + str(unsupported_layers))
            exit(1)
        ### TODO: Add any necessary extensions ###
        if extension is not None:
            self.core.add_extension(extension, device)

        self.input = next(iter(self.network.inputs))
        self.output = next(iter(self.network.outputs))

        ### TODO: Return the loaded inference plugin ###
        self.exec_network = self.core.load_network(self.network, device)
        return self.exec_network

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.network.inputs[self.input].shape

    def exec_net(self, image):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.start_async(request_id=0,
                                             inputs={self.input: image})

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[0].wait(-1)

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[0].outputs[self.output]

    def get_all_output(self):
        result = {}
        for i in list(self.network.outputs.keys()):
            result[i] = self.exec_network.requests[0].outputs[i]
        return result
