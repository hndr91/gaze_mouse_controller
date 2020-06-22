import os
import sys
from argparse import ArgumentParser
import cv2
import numpy as np
import constant
from input_feeder import InputFeeder
from model_face import ModelFace

def build_argparser():
    """
    Parse command line arguments

    :return: command line argumets
    """

    parser = ArgumentParser()
    parser.add_argument("-it", "--input_type", required=True, type=str, 
                        help="The type of input. Can be 'video' for video file, 'image' for image file, or 'cam' to use webcam feed.")
    parser.add_argument("-if", "--input_file", required=False, type=str,
                        help="The file that contains the input image or video file. Leave empty for cam input_type.")
    parser.add_argument("-d", "--device", type=str, default="CPU", 
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    # parser.add_argument("-l", "--cpu_extension", required=False, type=str, default=None,
    #                     help="MKLDNN (CPU)-targeted custom layers."
    #                          "Absolute path to a shared library with the"
    #                          "kernels impl. This is required for OpenVino 2019 and oldest")
    
    return parser


def infer_on_stream(args):
    # Get Input
    input_feeder = InputFeeder(args.input_type, args.input_file)
    input_feeder.load_data()

    # Load face detection model
    model_face = ModelFace(model_name=constant.FACE16, device=args.device, extensions=constant.CPU_EXTENSION)
    model_face.load_model()

    for frame in input_feeder.next_batch():
        # Break if number of next frame less then number of batch
        if frame is None:
            break

        _, output_frame = model_face.predict(frame)

        cv2.imshow('Capture', output_frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    input_feeder.close()
    cv2.destroyAllWindows()

def main():
    args = build_argparser().parse_args()
    infer_on_stream(args)

if __name__ == '__main__':
    main()