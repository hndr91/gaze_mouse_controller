import os
import sys
import cv2
from openvino.inference_engine import IECore, IENetwork

class HeadPoseEstimation:
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        Initiate class variables
        '''
        # path = os.getcwd()
        # root_path = os.path.abspath(os.path.join(path, os.pardir))
        self.model_xml = model_name + ".xml"
        self.model_bin = model_name + ".bin"
        self.device = device

        try:
            self.network = IENetwork(self.model_xml, self.model_bin)
        except Exception as e:
            print("Cannot initialize the network. Please enter correct model path. Error : ", e)

        self.core = IECore()

        # For OpenVino 2019 and older only
        if extensions and "CPU" in device:
            self.core.add_extension(extensions, self.device)


        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = iter(self.network.outputs)
        self.angle_p_fc = next(self.output_names)
        self.angle_r_fc = next(self.output_names)
        self.angle_y_fc = next(self.output_names)
        # self.output_shape = self.network.outputs[self.output_name].shape

    def load_model(self):
        '''
        Load model to the network
        '''
        try:
            self.exec_net = self.core.load_network(self.network, self.device)
        except Exception as e:
            print("Cannot load the model. Error : ", e)

    def predict(self, image):
        '''Estimate head pose based on pitch, rotation, and yaw
        
        Args:
            image: face image
        
        Returns:
            Array of yaw, pitch, rotation. Size (1x3)
        '''
        prep_image = self.preprocess_input(image)
        input_name = self.input_name
        
        input_dict = {input_name: prep_image}

        try:
            infer = self.exec_net.start_async(request_id=0, inputs=input_dict)
        except Exception as e:
            print("Cannot do inference. Error : ", e)

        status = infer.wait()

        if status == 0:
            angle_p_fc = infer.outputs[self.angle_p_fc][0][0]
            angle_r_fc = infer.outputs[self.angle_r_fc][0][0]
            angle_y_fc = infer.outputs[self.angle_y_fc][0][0]


        return [[angle_y_fc, angle_p_fc, angle_r_fc]]

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