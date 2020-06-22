import os
import sys
import cv2
from openvino.inference_engine import IECore, IENetwork

class ModelFace:
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


        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_name = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_name].shape

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

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
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
            outputs = infer.outputs[self.output_name]
            coords = self.preprocess_output(outputs)
            output_frame = self.draw_box(coords, image)


        return coords, output_frame

    def check_model(self):

        raise NotImplementedError

    def preprocess_input(self, image):
        n,c,h,w = self.input_shape
        prep_image = image
        prep_image = cv2.resize(prep_image, (w,h))
        prep_image = prep_image.transpose((2,0,1))
        prep_image = prep_image.reshape(n,c,h,w)

        return prep_image

    def preprocess_output(self, outputs):
        res = outputs[0][0]

        preds = [pred for pred in res if pred[1] == 1 and pred[2] > 0.5]

        coords = [[pred[3], pred[4], pred[5], pred[6]] for pred in preds]

        # This should be return cropped face
        return coords

    def draw_box(self, coords, image):
        h, w, _ = image.shape
        for coord in coords:
            x = int(coord[0] * w)
            y = int(coord[1] * h)
            x2 = int(coord[2] * w)
            y2 = int(coord[3] * h)

            cv2.rectangle(image, (x,y), (x2, y2), (255,0,0), 2)
        
        return image