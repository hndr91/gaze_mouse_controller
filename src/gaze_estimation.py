import os
import sys
import cv2
from openvino.inference_engine import IECore, IENetwork

class GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        path = os.getcwd()
        root_path = os.path.abspath(os.path.join(path, os.pardir))
        self.model_xml = root_path + model_name + ".xml"
        self.model_bin = root_path + model_name + ".bin"
        self.device = device

        try:
            self.network = IENetwork(self.model_xml, self.model_bin)
        except Exception as e:
            print("Cannot initialize the network. Please enter correct model path. Error : ", e)

        self.core = IECore()

        # For OpenVino 2019 and older only
        if extensions and "CPU" in device:
            self.core.add_extension(extensions, self.device)


        self.input_names = iter(self.network.inputs)
        self.head_pose = next(self.input_names)
        self.l_image = next(self.input_names)
        self.r_image = next(self.input_names)
        self.input_shape = self.network.inputs[self.l_image].shape
        self.output_name = next(iter(self.network.outputs))

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        try:
            self.exec_net = self.core.load_network(self.network, self.device)
        except Exception as e:
            print("Cannot load the model. Error : ", e)

    def predict(self, l_eye, r_eye, head_pose):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        prep_l_eye = self.preprocess_input(l_eye)
        prep_r_eye = self.preprocess_input(r_eye)
        head_pose_name = self.head_pose
        l_image_name = self.l_image
        r_image_name = self.r_image
        
        input_dict = {head_pose_name: head_pose, 
                     l_image_name: prep_l_eye, 
                     r_image_name: prep_r_eye}

        try:
            infer = self.exec_net.start_async(request_id=0, inputs=input_dict)
        except Exception as e:
            print("Cannot do inference. Error : ", e)

        status = infer.wait()

        if status == 0:
            outputs = infer.outputs[self.output_name]
            # coords = self.preprocess_output(outputs)
            # output_frame, cropped_face, box_coord = self.draw_box(coords, image)


        return outputs[0]

    def check_model(self):

        raise NotImplementedError

    def preprocess_input(self, image):
        n,c,h,w = self.input_shape
        prep_image = image
        prep_image = cv2.resize(prep_image, (w,h))
        prep_image = prep_image.transpose((2,0,1))
        prep_image = prep_image.reshape(n,c,h,w)

        return prep_image