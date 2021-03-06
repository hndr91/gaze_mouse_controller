import os
import sys
from argparse import ArgumentParser
import cv2
import numpy as np
import constant
from input_feeder import InputFeeder
from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from facial_landmark_detection import FacialLandmarkDetection
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController

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
    parser.add_argument("-l", "--cpu_extension", required=False, type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl. This is required for OpenVino 2019 and oldest")
    parser.add_argument("-p", "--precision", required=True, type=str, default="FP32",
                        help="Select model precision. It can be FP32, FP16, or INT8")
    parser.add_argument("-v", "--visualize", required=False, type=str, default="True",
                        help="Visualize or show the output frames." 
                             "Set to 'False' to turn it off or 'True to turn it on. Default='True'")
    
    return parser

def eyes_crop(image, landmark_x, landmark_y, crop_lenght=40):
    """Crop eyes images based on landmark.

    Crop eye region based on landmark and diagonal value of crop length for center point.
    The center poin is landmark point. The length of croped image is 2*(2*croped length).

    Args:
        image: face image from face detection model
        landmark_x: x point value from facial landmark detection model
        landmark_y: y point value from facial landmark detection model
        crop_length: diagonal value from center poin (default=40)

    Returns:
        coordinates of croped eye image, croped eye image
    """
    x = landmark_x - crop_lenght
    y = landmark_y - crop_lenght
    x2 = x + (2*crop_lenght)
    y2 = y + (2*crop_lenght)

    coords = [(x,y),(x2,y2)]
    cropped_eye = image[y:y2, x:x2].copy()

    return coords, cropped_eye

def select_precision(args, precision="FP32"):
    '''Select model precision

    This function let users to choose the model precision.
    Since face detection only support FP32-INT1 precision,
    it will used in all precision configuration. Therefore,
    HETERO with CPU failover should be use as the devices,
    if we want to utilized other devices except CPU.

    HETERO configuration also needed if you want to use 
    INT8 precision. Because, INT8 only supported in specific
    layers and device (ex: CPU)

    Example:
        HETERO:GPU,CPU
        HETERO:MYRIAD,CPU
    '''
    path = os.getcwd()
    root_path = os.path.abspath(os.path.join(path, os.pardir))


    if "FP32" in precision:
        face = root_path + constant.FACE32
        head = root_path + constant.HEAD32
        landmark = root_path + constant.LAND32
        gaze = root_path + constant.GAZE32

        return (face, head, landmark, gaze)
    elif "FP16" in precision:
        face = root_path + constant.FACE16
        head = root_path + constant.HEAD16
        landmark = root_path + constant.LAND16
        gaze = root_path + constant.GAZE16

        return (face, head, landmark, gaze)
    elif "INT8" in precision:
        face = root_path + constant.FACE8
        head = root_path + constant.HEAD8
        landmark = root_path + constant.LAND8
        gaze = root_path + constant.GAZE8

        return (face, head, landmark, gaze)
    else:
        raise Exception("{} is unrecognized precission".format(args.precision))


def infer_on_stream(args):
    models = None
    # Check selected precision model
    if "FP32" in args.precision:
        models = select_precision(args.precision)

    if "FP16" in args.precision:
        models = select_precision(args.precision)

    if "INT8" in args.precision:
        models = select_precision(args.precision)


    # Get Input
    input_feeder = InputFeeder(args.input_type, args.input_file)
    input_feeder.load_data()

    # Load face detection model
    face = FaceDetection(model_name=models[0], device=args.device, extensions=args.cpu_extension)
    face.load_model()

    # Load head pose model
    head = HeadPoseEstimation(model_name=models[1], device=args.device, extensions=args.cpu_extension)
    head.load_model()

    # Load facial landmark model
    landmark = FacialLandmarkDetection(model_name=models[2], device=args.device, extensions=args.cpu_extension)
    landmark.load_model()

    # Load gaze estimation model
    gaze = GazeEstimation(model_name=models[3], device=args.device, extensions=args.cpu_extension)
    gaze.load_model()

    # Initalize mouse controller
    mouse = MouseController('high', 'fast')

    for frame in input_feeder.next_batch():
        # Break if number of next frame less then number of batch
        if frame is None:
            break

        # Estimate face region
        output_frame, cropped_face, box_coord = face.predict(frame)
        
        # Estimate head pose position
        head_pose = head.predict(cropped_face)
        head_pose = np.array(head_pose)
        
        # Estimate eyes landmark coordinates
        lr_eyes = landmark.predict(cropped_face)

        eyes = []

        # Calculate eye image region
        for coord in lr_eyes:
            x = int(coord[0] + box_coord[0])
            y = int(coord[1] + box_coord[1])
            cv2.circle(output_frame, (x, y), 5, (255,0,0), -1)

            eye_box, cropped_eye = eyes_crop(output_frame, x, y, 40)
            cv2.rectangle(output_frame, eye_box[0], eye_box[1], (255,0,0), 1)
            eyes.append(cropped_eye)
        
        # Estimate gaze direction
        gaze_coords = gaze.predict(eyes[0], eyes[1], head_pose)

        # Move the mouse cursor
        mouse.move(gaze_coords[0], gaze_coords[1])
        
        if "True" in args.visualize:
            cv2.imshow('Capture', output_frame)
        
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    input_feeder.close()
    if "True" in args.visualize:
        cv2.destroyAllWindows()

def main():
    args = build_argparser().parse_args()
    infer_on_stream(args)

if __name__ == '__main__':
    main()