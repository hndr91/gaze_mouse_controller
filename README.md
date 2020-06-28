# Computer Pointer Controller
This project demonstrate the ability to use Intel OpenVino IR model to control computer's pointer. In this project, 4 IR model are utilized. The inference pipeline is showns below

![inference pipeline](img/pipeline.png)

The input images comes from video, image, or webcam. This project also ultized PyAutoGui to control the pointer as the inference result. You also need Intel OpenVino R3.1 or later.

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

### Supported Hardwares
To able run the application, your system should meet at least one of the following hardwares.

- 6th gen Intel Core family processor
- Intel Xeon processor
- Intel Xeon E family
- Intel Atom processor with support for Intel SSE4.1
- Intel Pentium processor N4200/5, N3350/5, or N3450/5 with Intel HD Graphic
- Intel FPGA
- Intel Movidious VPU

### Supported OS
- Windows 10 (not tested)
- Ubuntu 18.04 LTS (64bit)
- CentOS (not tested)
- macOS 10.13 (not tested)

### Dependencies
- image==1.5.27
- ipdb==0.12.3
- ipython==7.10.2
- numpy==1.17.4
- Pillow==6.2.1
- requests==2.22.0
- virtualenv==16.7.9

### Install Dependencies
`$ pip install -r requirements.txt`


### Models Directory
By default this project using the following models

- face-detection-adas-binary-0001
- head-pose-estimation-adas-0001
- landmarks-regression-retail-0009
- gaze-estimation-adas-0002

If you want to use other models, please put it to `models` directory.

for example:
```
models/<your_models>/<precision>/your_model.xml
models/<your_models>/<precision>/your_model.bin
```

You also need to update the `src/constant.py` to follow your models directory


## Demo
Go to `src` directory and run this command
`$ python main.py -it video -if ../bin/video -p FP32 -l <path to CPU extension>`

If you use OpenVino 2020.1 or later
`$ python main.py -it video -if ../bin/video -p FP32`


## Documentation
```
usage: main.py [-it] [-if] [-d] [-l] [-p]

General:
    -h, --help          show the help

arguments:
    -it, --input_type   project input type. It can be video, image, or webcam
    -if, --input_file   directory of input file. This is required for video and image
    -d, --device        the device to infer on. CPU, GPU, MYRIAD, FPGA
    -l, --cpu_extension the path to cpu extension. Required for OpenVino 2019 and older
    -p, --precision     model precision option. FP32, FP16, or INT8

example:
    main.py -it video -if video.mp4 -p FP32
    main.py -it video -if video.mp4 -d CPU -l libcpu_extension_sse4.so -p FP32
```

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
