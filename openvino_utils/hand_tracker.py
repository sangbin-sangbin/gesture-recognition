import os
import re
import sys
import time

import cv2
import numpy as np
import openvino.runtime as ov
import yaml

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from openvino_utils import mediapipe_utils as mpu

with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)


class HandTracker:
    def __init__(
        self,
        input_src=None,
        pd_xml="mediapipe_models/palm_detection_FP16.xml",
        pd_device="CPU",
        pd_score_thresh=0.6,
        pd_nms_thresh=0.3,
        lm_xml="mediapipe_models/hand_landmark_FP16.xml",
        lm_device="CPU",
        lm_score_threshold=0.6,
    ):
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.lm_score_threshold = lm_score_threshold

        self.dataset = []
        self.state = "break"
        self.start_time = time.time()
        self.gesture_num = 0
        self.gestures = config["gestures"]

        self.recognized = []

        if input_src.endswith(".jpg") or input_src.endswith(".png"):
            self.image_mode = True
            self.img = cv2.imread(input_src)
        else:
            self.image_mode = False
            if input_src.isdigit():
                input_src = int(input_src)
            # Open a webcam
            w = 1280
            h = 720
            self.cap = cv2.VideoCapture(input_src , cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        # Create SSD anchors
        # https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt
        anchor_options = mpu.SSDAnchorOptions(
            num_layers=4,
            min_scale=0.1484375,
            max_scale=0.75,
            input_size_height=192,
            input_size_width=192,
            anchor_offset_x=0.5,
            anchor_offset_y=0.5,
            strides=[8, 16, 16, 16],
            aspect_ratios=[1.0],
            reduce_boxes_in_lowest_layer=False,
            interpolated_scale_aspect_ratio=1.0,
            fixed_anchor_size=True,
        )
        self.anchors = mpu.generate_anchors(anchor_options)
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

        # Load Openvino models
        self.load_models(pd_xml, pd_device, lm_xml, lm_device)

    # Getter method
    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

    # Setter method
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def load_models(self, pd_xml, pd_device, lm_xml, lm_device):
        print("Loading Inference Engine")
        self.core = ov.Core()
        print("Device info:")
        versions = self.core.get_versions(pd_device)
        print(f"{' ' * 8}{pd_device}")
        print(
            f"{' ' * 8}MKLDNNPlugin version ......... {versions[pd_device].major}.{versions[pd_device].minor}"
        )
        print(
            f"{' ' * 8}Build ........... {versions[pd_device].build_number}"
        )

        # Pose detection model
        pd_name = os.path.splitext(pd_xml)[0]
        pd_bin = pd_name + ".bin"
        print(
            f"Palm Detection model - Reading network files:\n\t{pd_xml}\n\t{pd_bin}"
        )
        self.pd_model = self.core.read_model(model=pd_xml, weights=pd_bin)
        # Input blob: input_1 - shape: [1, 3, 192, 192]
        # Output blob: Identity - shape: [1, 2016, 18]
        # Output blob: Identity_1 - shape: [1, 2016, 1]
        self.pd_input_blob = next(iter(next(iter(self.pd_model.inputs)).names))
        input_shape = next(iter(self.pd_model.inputs)).shape
        print(f"Input blob: {self.pd_input_blob} - shape: {input_shape}")
        _, _, self.pd_h, self.pd_w = next(iter(self.pd_model.inputs)).shape

        pattern = re.compile(r"^(?!.*:)(?:Identity.*$|.*\/BiasAdd\/Add$)")

        for o in self.pd_model.outputs:
            output_names = list(o.names)
            for name in output_names:
                if pattern.match(name):
                    print(f"Output blob: {name} - shape: {list(o.shape)}")
        self.pd_scores = "Identity_1"
        self.pd_bboxes = "Identity"
        print("Loading palm detection model into the plugin")
        self.pd_exec_model = self.core.compile_model(
            model=self.pd_model, device_name=pd_device
        )

        # Landmarks model
        if lm_device != pd_device:
            print("Device info:")
            versions = self.core.get_versions(pd_device)
            print(f"{' ' * 8}{pd_device}")
            print(
                f"{' ' * 8}MKLDNNPlugin version ......... {versions[pd_device].major}.{versions[pd_device].minor}"
            )
            print(
                f"{' ' * 8}Build ........... {versions[pd_device].build_number}"
            )

        lm_name = os.path.splitext(lm_xml)[0]
        lm_bin = lm_name + ".bin"
        print(
            f"Landmark model - Reading network files:\n\t{lm_xml}\n\t{lm_bin}"
        )
        self.lm_model = self.core.read_model(model=lm_xml, weights=lm_bin)
        # Input blob: input_1 - shape: [1, 3, 224, 224]
        # Output blob: Identity_1 - shape: [1, 1]
        # Output blob: Identity_2 - shape: [1, 1]
        # Output blob: Identity_3_dense/BiasAdd/Add - shape: [1, 63]
        # Output blob: Identity_dense/BiasAdd/Add - shape: [1, 63]
        self.lm_input_blob = next(iter(next(iter(self.lm_model.inputs)).names))
        input_shape = next(iter(self.pd_model.inputs)).shape
        print(f"Input blob: {self.pd_input_blob} - shape: {input_shape}")

        _, _, self.lm_h, self.lm_w = next(iter(self.lm_model.inputs)).shape
        for o in self.lm_model.outputs:
            output_names = list(o.names)
            for name in output_names:
                if pattern.match(name):
                    print(f"Output blob: {name} - shape: {list(o.shape)}")
        self.lm_score = "Identity_1"
        self.lm_handedness = "Identity_2"
        self.lm_landmarks = "Identity_dense/BiasAdd/Add"
        print("Loading landmark model to the plugin")
        self.lm_exec_model = self.core.compile_model(
            model=self.lm_model, device_name=lm_device
        )

    def pd_postprocess(self, inference, frame_size):
        scores = np.squeeze(inference[self.pd_scores])  # 2016
        bboxes = inference[self.pd_bboxes][0]  # 2016x18
        # Decode bboxes
        regions = mpu.decode_bboxes(
            self.pd_score_thresh, scores, bboxes, self.anchors
        )
        # Non maximum suppression
        regions = mpu.non_max_suppression(
            regions,
            self.pd_nms_thresh
        )
        regions = mpu.detections_to_rect(regions)
        regions = mpu.rect_transformation(
            regions,
            frame_size,
            frame_size
        )
        return regions

    def lm_postprocess(self, region, inference):
        region.lm_score = np.squeeze(inference[self.lm_score])
        region.handedness = np.squeeze(inference[self.lm_handedness])
        lm_raw = np.squeeze(inference[self.lm_landmarks])

        lm = []
        for i in range(int(len(lm_raw) / 3)):
            # x,y,z -> x/w,y/h,z/w (here h=w)
            lm.append(lm_raw[3 * i: 3 * (i + 1)] / self.lm_w)
        region.landmarks = lm
        return region

    def lm_xy_coordinates(self, region):
        src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
        dst = np.array(
            [(x, y) for x, y in region.rect_points[1:]], dtype=np.float32
        )  # region.rect_points[0] is left bottom point !
        mat = cv2.getAffineTransform(src, dst)
        lm_xy = np.expand_dims(
            np.array([(lm[0], lm[1]) for lm in region.landmarks]), axis=0
        )
        lm_xy = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int32)
        return lm_xy

    def inference(self, frame):
        h, w = frame.shape[:2]

        # Padding on the small side to get a square shape
        frame_size = max(h, w)
        pad_h = int((frame_size - h) / 2)
        pad_w = int((frame_size - w) / 2)
        frame = cv2.copyMakeBorder(
            frame, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT
        )

        # Resize image to NN square input shape
        frame_nn = cv2.resize(
            frame, (self.pd_w, self.pd_h), interpolation=cv2.INTER_AREA
        )

        # Transpose hxwx3 -> 1x3xhxw
        frame_nn = np.transpose(frame_nn, (2, 0, 1))[None,]

        # Get palm detection
        pd_infer_request = self.pd_exec_model.create_infer_request()
        inference = pd_infer_request.infer(
            inputs={self.pd_input_blob: frame_nn}
        )
        regions = self.pd_postprocess(inference, frame_size)

        # Hand landmarks
        results = []
        for region in regions:
            frame_nn = mpu.warp_rect_img(
                region.rect_points, frame, self.lm_w, self.lm_h
            )
            # Transpose hxwx3 -> 1x3xhxw
            frame_nn = np.transpose(frame_nn, (2, 0, 1))[None,]

            # Get hand landmarks
            lm_infer_request = self.lm_exec_model.create_infer_request()
            inference = lm_infer_request.infer(
                inputs={self.lm_input_blob: frame_nn}
            )
            region = self.lm_postprocess(region, inference)
            results.append({"handedness": region.handedness, "landmark": self.lm_xy_coordinates(region)})
        return results
