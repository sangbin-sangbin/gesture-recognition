import numpy as np
from multiprocessing import Process
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)
import cv2
import glob
import time
from config import imagenet_classes


# The target can be used as a context manager (”with” statement) to ensure it's released on time.
# Here it's avoided for the sake of simplicity
target = VDevice()

# Loading compiled HEFs to device:
model_name = 'resnet_v1_50'
hef_path = f'./hefs/{model_name}.hef'
hef = HEF(hef_path)

# Configure network groups
configure_params = ConfigureParams.create_from_hef(hef=hef,interface=HailoStreamInterface.PCIe)
network_groups = target.configure(hef, configure_params)
network_group = network_groups[0]
network_group_params = network_group.create_params()

# Create input and output virtual streams params
# Quantized argument signifies whether or not the incoming data is already quantized.
# Data is quantized by HailoRT if and only if quantized == False .
input_vstreams_params = InputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)
output_vstreams_params = OutputVStreamParams.make(network_group, quantized=True, format_type=FormatType.UINT8)

# Define dataset params
input_vstream_info = hef.get_input_vstream_infos()[0]
output_vstream_info = hef.get_output_vstream_infos()[0]
image_height, image_width, channels = input_vstream_info.shape
batch_size = 1

# Load dataset
names = files = glob.glob('./dataset/imagenet/*')

dataset = []
for i, name in enumerate(names):
    img = cv2.imread(name)
    img = cv2.resize(img, dsize=(image_height, image_width), interpolation=cv2.INTER_LINEAR)
    dataset.append(img)
dataset = np.array(dataset).astype(np.float32)

infer_sum = 0
infer_num = 0

# Infer
dataset_len = len(dataset)
i = 0
while i + batch_size < dataset_len:
    input_dataset = dataset[i: i+batch_size]
    i += batch_size
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        input_data = {input_vstream_info.name: input_dataset}
        with network_group.activate(network_group_params):
            start = time.time_ns() // 1000000
            infer_results = infer_pipeline.infer(input_data)
            infer_sum += time.time_ns() // 1000000 - start
            infer_num += 1

            for idx, res in enumerate(infer_results[output_vstream_info.name]):
                text = imagenet_classes[np.argmax(res)]
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                result_img = cv2.resize(input_dataset[idx] / 255, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
                result_img = cv2.putText(result_img, text, (10, text_size[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.imshow('', result_img)
                cv2.waitKey(0)

print('average inference time:', infer_sum / infer_num)