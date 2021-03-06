import os
import sys
import cv2
import logging
logging.basicConfig(level=logging.INFO)
from openvino.inference_engine import IECore, IENetwork

class FacialLandmarkDetection:
    '''
    Class for the Facial landmark Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        Initiate class variables
        '''
        self.model_xml = model_name + ".xml"
        self.model_bin = model_name + ".bin"
        self.device = device

        try:
            self.network = IENetwork(self.model_xml, self.model_bin)
        except Exception as e:
            logging.info("Cannot initialize the network. Please enter correct model path. Error : %s", e)

        self.core = IECore()

        # For OpenVino 2019 and older only
        if extensions and "HETERO" in device and "CPU" in device:
            self.core.add_extension(extensions, "CPU")

        if extensions and "CPU" in device and not "HETERO" in device:
            self.core.add_extension(extensions, self.device)


        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_name = next(iter(self.network.outputs))
        # self.output_shape = self.network.outputs[self.output_name].shape

    def load_model(self):
        '''
        Load model to the network
        '''
        try:
            self.exec_net = self.core.load_network(self.network, self.device)
        except Exception as e:
            logging.info("Cannot load the model. Error : %s", e)

    def predict(self, image):
        '''Estimate eyes landmark based on face image

        Args:
            image: face image
        
        Returns:
            Array of eyes landmark coordintes based on catersian coordinates
        '''
        prep_image = self.preprocess_input(image)
        input_name = self.input_name
        
        input_dict = {input_name: prep_image}

        try:
            infer = self.exec_net.start_async(request_id=0, inputs=input_dict)
        except Exception as e:
            logging.info("Cannot do inference. Error : %s", e)

        status = infer.wait()

        if status == 0:
            outputs = infer.outputs[self.output_name]
            coords = self.preprocess_output(outputs)
            scaled_coords = self.scaled_coords(coords, image)

        return scaled_coords

    def check_model(self):

        raise NotImplementedError

    def preprocess_input(self, image):
        '''Input image preprocessing.

        Args:
            image: original image
        
        Returns:
            preprocessed image based on model input tensor
        '''
        n,c,h,w = self.input_shape
        prep_image = image
        prep_image = cv2.resize(prep_image, (w,h))
        prep_image = prep_image.transpose((2,0,1))
        prep_image = prep_image.reshape(n,c,h,w)

        return prep_image

    def preprocess_output(self, outputs):
        '''Get left eye and right eye landmark coordinates only.

        Args:
            outputs: facial landmark detection model output
        
        Returns:
            Array of eyes coordinates based on catertesian coordinates
        '''
        res = outputs.ravel()
        l_eye = [res[0], res[1]]
        r_eye = [res[2], res[3]]

        return [l_eye, r_eye]

    def scaled_coords(self, coords, image):
        '''Translate coordinate to original image size

        Args:
            coords: array of coorditates of left and right eyes landmark
            image: original face image
        
        Returns:
            Array of eyes translated coordinates
        '''
        h, w, _ = image.shape
        scaled = []
        for coord in coords:
            x = int(coord[0] * w)
            y = int(coord[1] * h)
            scaled.append([x,y])
        
        return scaled