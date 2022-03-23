from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import cv2
import time
import datetime
import sys

#from qarpo.demoutils import *

import logging as log
import argparse

import io
import argparse
import os
from openvino.inference_engine import IECore
import threading


print('Imported Python modules successfully.')

def read_data(input_data):
    npzfiles = np.load(input_data, allow_pickle = True )
    array_01, array_02 = npzfiles.files
    img_batch = npzfiles[array_01]
    lbl_batch = npzfiles[array_02]
    images = np.vstack(img_batch)
    labels = np.hstack(lbl_batch)
    return(images, labels)

def build_argparser():
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--device", required=False,
                    default='CPU', help="Device type")
    ap.add_argument("-i", "--input_data", required=False,
                    default='test_data/testdata.npz', help="Input data file")
    ap.add_argument("-m", "--model", required=False,
                    default='FP32', help="Model type")
    ap.add_argument("-o", "--output", required=False,
                    default='results/', help="Output directory")
    ap.add_argument("-n", "--num_instance", required=False,
                    default="1", help="Number of inference instances")
    return ap

class InferReqWrap:
    def __init__(self, request, id, num_iter):
        self.id = id
        self.request = request
        self.num_iter = num_iter
        self.cur_iter = 0
        self.cv = threading.Condition()
        self.request.set_completion_callback(self.callback, self.id)

    def callback(self, statusCode, userdata):
        if (userdata != self.id):
            log.error("Request ID {} does not correspond to user data {}".format(self.id, userdata))
        elif statusCode != 0:
            log.error("Request {} failed with status code {}".format(self.id, statusCode))
        self.cur_iter += 1
        if self.cur_iter < self.num_iter:
            # here a user can read output containing inference results and put new input
            # to repeat async request again
            self.request.async_infer(self.input)
        else:
            # continue sample execution after last Asynchronous inference request execution
            self.cv.acquire()
            self.cv.notify()
            self.cv.release()

    def execute(self, mode, input_data):
        if (mode == "async"):
            self.input = input_data
            self.request.async_infer(input_data)
            self.cv.acquire()
            self.cv.wait()
            self.cv.release()
        elif (mode == "sync"):
            for self.cur_iter in range(self.num_iter):
                # here we start inference synchronously and wait for
                # last inference request execution
                self.request.infer(input_data)
        else:
            log.error("wrong inference mode is chosen. Please use \"sync\" or \"async\" mode")
            sys.exit(1)

def main():
    # Set up logging
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    args = vars(build_argparser().parse_args())
    print(args)
    # Arguments
    device_type = args['device']
    input_data = args['input_data']
    fp_model = args['model']
    output = args['output']
    num_instance = int(args['num_instance'])

    log.info("Creating Inference Engine")
    ie = IECore()

    # Get the input video from the specified path
    log.info("Reading data from {}".format(input_data))
    images, labels = read_data(input_data)

    pred_labels = np.zeros(labels.shape)
    num_samples = images.shape[0]

    # Set up OpenVINO inference
    log.info(f"Loading network:\n\t{fp_model}")
    net = ie.read_network(model='./models/ov/' + fp_model + '/saved_model.xml', \
                          weights='./models/ov/' + fp_model + '/saved_model.bin')

    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    net.batch_size = 1

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=device_type, num_requests=num_instance)
    n, c, h, w = net.input_info[input_blob].input_data.shape

    print("N, C, H, W:", n,c,h,w)

    # Start inference request execution. Wait for last execution being completed
    for ind in range(0, num_samples, num_instance):
        if ind+num_instance >= num_samples:
            num_instance = num_samples - ind

        image_test = np.transpose(images[ind:ind+num_instance, :,:,:], (0,3,1,2))
        if num_instance == 1:
            # create one inference request for asynchronous execution
            request_id = 0
            infer_request = exec_net.requests[request_id]
            request_wrap = InferReqWrap(infer_request, request_id, 1)
            request_wrap.execute("sync", {input_blob: image_test})
            res = infer_request.output_blobs[out_blob]
            for _, probs in enumerate(res.buffer):
                    probs = np.squeeze(probs)
                    pred_labels[ind] = np.argmax(probs)
                    log.info(f"Current {ind}th class label is {np.argmax(probs)}")
        else:
            infer_request = []
            for i in range(num_instance):
                # create one inference request for asynchronous execution
                request_id = i
                infer_request.append(exec_net.requests[request_id])
                request_wrap = InferReqWrap(infer_request[-1], request_id, 1)
                request_wrap.execute("async", {input_blob: np.expand_dims(image_test[i,:,:,:], 0)})

            for i in range(num_instance):
                # Processing output blob
                res = infer_request[i].output_blobs[out_blob]
                for _, probs in enumerate(res.buffer):
                    probs = np.squeeze(probs)
                    pred_labels[ind+i] = np.argmax(probs)
                    log.info(f"Current {ind + i}th sample's class label is {np.argmax(probs)}")

    output_file = os.path.join(output, 'stats'+'.txt')
    log.info(f"Write the prediction into the {output_file} file.")
    with open(output_file, 'w') as f:
        for i in range(num_samples):
            f.write(str(i) + ", ")
            f.write(str(labels[i])+', ')
            f.write(str(int(pred_labels[i]))+'\n')
    f.close()


if __name__ == '__main__':
    sys.exit(main() or 0)
