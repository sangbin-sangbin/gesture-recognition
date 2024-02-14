import numpy as np
import mediapipe_utils as mpu
import cv2
from FPS import FPS, now
import os
import time
from openvino.inference_engine import IECore


class HandTracker:
    def __init__(
        self,
        input_src=None,
        pd_xml="mediapipe_models/palm_detection_FP32.xml",
        pd_device="CPU",
        pd_score_thresh=0.5,
        pd_nms_thresh=0.3,
        use_lm=True,
        lm_xml="mediapipe_models/hand_landmark_FP32.xml",
        lm_device="CPU",
        lm_score_threshold=0.5,
        use_gesture=False,
        crop=False,
        is_getdata=False,
    ):

        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.use_lm = use_lm
        self.lm_score_threshold = lm_score_threshold
        self.use_gesture = use_gesture
        self.crop = crop

        self.is_getdata = is_getdata
        self.dataset = []
        self.coords = []
        self.state = "break"
        self.start_time = time.time()
        self.gesture_num = 0
        self.gestures = ["default", "left", "right", "select", "exit", "none"]

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
            self.cap = cv2.VideoCapture(input_src)
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

        # Rendering flags
        if self.use_lm:
            self.show_pd_box = False
            self.show_pd_kps = False
            self.show_rot_rect = False
            self.show_handedness = False
            self.show_landmarks = True
            self.show_scores = False
        else:
            self.show_pd_box = True
            self.show_pd_kps = True
            self.show_rot_rect = False
            self.show_scores = False

    # Getter method
    def __getattribute__(self, name):
        if name == "dataset":
            return object.__getattribute__(self, "dataset")
        else:
            return object.__getattribute__(self, name)

    # Setter method
    def __setattr__(self, name, value):
        if name == "dataset":
            object.__setattr__(self, "dataset", value)
        else:
            object.__setattr__(self, name, value)

    def load_models(self, pd_xml, pd_device, lm_xml, lm_device):

        print("Loading Inference Engine")
        self.ie = IECore()
        print("Device info:")
        versions = self.ie.get_versions(pd_device)
        print("{}{}".format(" " * 8, pd_device))
        print(
            "{}MKLDNNPlugin version ......... {}.{}".format(
                " " * 8, versions[pd_device].major, versions[pd_device].minor
            )
        )
        print("{}Build ........... {}".format(" " * 8, versions[pd_device].build_number))

        # Pose detection model
        pd_name = os.path.splitext(pd_xml)[0]
        pd_bin = pd_name + ".bin"
        print("Palm Detection model - Reading network files:\n\t{}\n\t{}".format(pd_xml, pd_bin))
        self.pd_net = self.ie.read_network(model=pd_xml, weights=pd_bin)
        # Input blob: input_1 - shape: [1, 3, 192, 192]
        # Output blob: Identity - shape: [1, 2016, 18]
        # Output blob: Identity_1 - shape: [1, 2016, 1]
        self.pd_input_blob = next(iter(self.pd_net.input_info))
        print(
            f"Input blob: {self.pd_input_blob} - shape: {self.pd_net.input_info[self.pd_input_blob].input_data.shape}"
        )
        _, _, self.pd_h, self.pd_w = self.pd_net.input_info[self.pd_input_blob].input_data.shape

        for o in self.pd_net.outputs.keys():
            print(f"Output blob: {o} - shape: {self.pd_net.outputs[o].shape}")
            self.pd_scores = "Identity_1"
            self.pd_bboxes = "Identity"
        print("Loading palm detection model into the plugin")
        self.pd_exec_net = self.ie.load_network(network=self.pd_net, num_requests=1, device_name=pd_device)
        self.pd_infer_time_cumul = 0
        self.pd_infer_nb = 0

        self.infer_nb = 0
        self.infer_time_cumul = 0

        # Landmarks model
        if self.use_lm:
            if lm_device != pd_device:
                print("Device info:")
                versions = self.ie.get_versions(pd_device)
                print("{}{}".format(" " * 8, pd_device))
                print(
                    "{}MKLDNNPlugin version ......... {}.{}".format(
                        " " * 8, versions[pd_device].major, versions[pd_device].minor
                    )
                )
                print("{}Build ........... {}".format(" " * 8, versions[pd_device].build_number))

            lm_name = os.path.splitext(lm_xml)[0]
            lm_bin = lm_name + ".bin"
            print("Landmark model - Reading network files:\n\t{}\n\t{}".format(lm_xml, lm_bin))
            self.lm_net = self.ie.read_network(model=lm_xml, weights=lm_bin)
            # Input blob: input_1 - shape: [1, 3, 224, 224]
            # Output blob: Identity_1 - shape: [1, 1]
            # Output blob: Identity_2 - shape: [1, 1]
            # Output blob: Identity_3_dense/BiasAdd/Add - shape: [1, 63]
            # Output blob: Identity_dense/BiasAdd/Add - shape: [1, 63]
            self.lm_input_blob = next(iter(self.lm_net.input_info))
            print(
                f"Input blob: {self.lm_input_blob} - shape: {self.lm_net.input_info[self.lm_input_blob].input_data.shape}"
            )
            _, _, self.lm_h, self.lm_w = self.lm_net.input_info[self.lm_input_blob].input_data.shape
            # Batch reshaping if lm_2 is True
            for o in self.lm_net.outputs.keys():
                print(f"Output blob: {o} - shape: {self.lm_net.outputs[o].shape}")
            self.lm_score = "Identity_1"
            self.lm_handedness = "Identity_2"
            self.lm_landmarks = "Identity_dense/BiasAdd/Add"
            print("Loading landmark model to the plugin")
            self.lm_exec_net = self.ie.load_network(network=self.lm_net, num_requests=1, device_name=lm_device)
            self.lm_infer_time_cumul = 0
            self.lm_infer_nb = 0
            self.lm_hand_nb = 0

    def pd_postprocess(self, inference):
        scores = np.squeeze(inference[self.pd_scores])  # 2016
        bboxes = inference[self.pd_bboxes][0]  # 2016x18
        # Decode bboxes
        self.regions = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors)
        # Non maximum suppression
        self.regions = mpu.non_max_suppression(self.regions, self.pd_nms_thresh)
        if self.use_lm:
            mpu.detections_to_rect(self.regions)
            mpu.rect_transformation(self.regions, self.frame_size, self.frame_size)

    def pd_render(self, frame):
        for r in self.regions:
            if self.show_pd_box:
                box = (np.array(r.pd_box) * self.frame_size).astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
            if self.show_pd_kps:
                for i, kp in enumerate(r.pd_kps):
                    x = int(kp[0] * self.frame_size)
                    y = int(kp[1] * self.frame_size)
                    cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
                    cv2.putText(frame, str(i), (x, y + 12), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            if self.show_scores:
                cv2.putText(
                    frame,
                    f"Palm score: {r.pd_score:.2f}",
                    (int(r.pd_box[0] * self.frame_size + 10), int((r.pd_box[1] + r.pd_box[3]) * self.frame_size + 60)),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 255, 0),
                    2,
                )

    def lm_postprocess(self, region, inference):
        region.lm_score = np.squeeze(inference[self.lm_score])
        region.handedness = np.squeeze(inference[self.lm_handedness])
        lm_raw = np.squeeze(inference[self.lm_landmarks])

        lm = []
        for i in range(int(len(lm_raw) / 3)):
            # x,y,z -> x/w,y/h,z/w (here h=w)
            lm.append(lm_raw[3 * i : 3 * (i + 1)] / self.lm_w)
        region.landmarks = lm
        self.lanmark_list = [[float(coords[0]), float(coords[1])] for coords in region.landmarks]
        # self.recognized.append(self.lanmark_list)

    def lm_render(self, frame, region):
        if region.lm_score > self.lm_score_threshold:
            if self.is_getdata:
                # Store the lnadmarks in dataset when recording
                if self.state == "recording":
                    self.dataset.append(
                        {"landmarks": mpu.normalize_points(self.lanmark_list), "gesture": self.gesture_num}
                    )

                if (time.time() - self.start_time) > 30 and self.state == "recording":
                    self.start_time = time.time()
                    self.state = "break"
                    self.gesture_num = (self.gesture_num + 1) % len(self.gestures)
                elif (time.time() - self.start_time) > 10 and self.state == "break":
                    self.start_time = time.time()
                    self.state = "recording"

                if self.state == "break":
                    cv2.putText(
                        frame,
                        'Break.. Next gesture is "' + self.gestures[self.gesture_num] + '"',
                        (frame.shape[1] // 2 - 430, frame.shape[0] // 2 - 220),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 0, 0),
                        3,
                    )
                else:
                    cv2.putText(
                        frame,
                        'Recorging "' + self.gestures[self.gesture_num] + '"',
                        (frame.shape[1] // 2 - 430, frame.shape[0] // 2 - 220),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 0, 0),
                        3,
                    )
            else:
                self.dataset.append({"landmarks": mpu.normalize_points(self.lanmark_list), "gesture": self.gesture_num})

            if self.show_rot_rect:
                cv2.polylines(frame, [np.array(region.rect_points)], True, (0, 255, 255), 2, cv2.LINE_AA)

            if self.show_landmarks:
                self.landmarks = []

                src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
                dst = np.array(
                    [(x, y) for x, y in region.rect_points[1:]], dtype=np.float32
                )  # region.rect_points[0] is left bottom point !
                mat = cv2.getAffineTransform(src, dst)
                lm_xy = np.expand_dims(np.array([(l[0], l[1]) for l in region.landmarks]), axis=0)
                lm_xy = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int32)
                list_connections = [
                    [0, 1, 2, 3, 4],
                    [0, 5, 6, 7, 8],
                    [5, 9, 10, 11, 12],
                    [9, 13, 14, 15, 16],
                    [13, 17],
                    [0, 17, 18, 19, 20],
                ]
                lines = [np.array([lm_xy[point] for point in line]) for line in list_connections]
                cv2.polylines(frame, lines, False, (255, 0, 0), 2, cv2.LINE_AA)
                self.landmarks.append(lm_xy)
                if self.is_getdata:
                    for x, y in lm_xy:
                        cv2.circle(frame, (x, y), 6, (0, 128, 255), -1)

            if self.show_handedness:
                cv2.putText(
                    frame,
                    f"RIGHT {region.handedness:.2f}" if region.handedness > 0.5 else f"LEFT {1-region.handedness:.2f}",
                    (
                        int(region.pd_box[0] * self.frame_size + 10),
                        int((region.pd_box[1] + region.pd_box[3]) * self.frame_size + 20),
                    ),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0) if region.handedness > 0.5 else (0, 0, 255),
                    2,
                )
            if self.show_scores:
                cv2.putText(
                    frame,
                    f"Landmark score: {region.lm_score:.2f}",
                    (
                        int(region.pd_box[0] * self.frame_size + 10),
                        int((region.pd_box[1] + region.pd_box[3]) * self.frame_size + 90),
                    ),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 255, 0),
                    2,
                )

    def run(self):

        self.fps = FPS(mean_nb_frames=20)

        nb_pd_inferences = 0
        nb_lm_inferences = 0
        glob_pd_rtrip_time = 0
        glob_lm_rtrip_time = 0
        while True:
            self.fps.update()
            if self.image_mode:
                vid_frame = self.img
            else:
                ok, vid_frame = self.cap.read()
                if not ok:
                    break
            h, w = vid_frame.shape[:2]
            if self.crop:
                # Cropping the long side to get a square shape
                self.frame_size = min(h, w)
                dx = (w - self.frame_size) // 2
                dy = (h - self.frame_size) // 2
                video_frame = vid_frame[dy : dy + self.frame_size, dx : dx + self.frame_size]
            else:
                # Padding on the small side to get a square shape
                self.frame_size = max(h, w)
                pad_h = int((self.frame_size - h) / 2)
                pad_w = int((self.frame_size - w) / 2)
                video_frame = cv2.copyMakeBorder(vid_frame, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT)

            # Resize image to NN square input shape
            frame_nn = cv2.resize(video_frame, (self.pd_w, self.pd_h), interpolation=cv2.INTER_AREA)

            # Transpose hxwx3 -> 1x3xhxw
            frame_nn = np.transpose(frame_nn, (2, 0, 1))[None,]

            annotated_frame = video_frame.copy()

            # Get palm detection
            pd_rtrip_time = now()
            inference = self.pd_exec_net.infer(inputs={self.pd_input_blob: frame_nn})
            glob_pd_rtrip_time += now() - pd_rtrip_time
            self.pd_postprocess(inference)
            self.pd_render(annotated_frame)
            nb_pd_inferences += 1

            # Hand landmarks
            if self.use_lm:
                for i, r in enumerate(self.regions):
                    frame_nn = mpu.warp_rect_img(r.rect_points, video_frame, self.lm_w, self.lm_h)
                    # Transpose hxwx3 -> 1x3xhxw
                    frame_nn = np.transpose(frame_nn, (2, 0, 1))[None,]

                    # Get hand landmarks
                    lm_rtrip_time = now()
                    inference = self.lm_exec_net.infer(inputs={self.lm_input_blob: frame_nn})
                    glob_lm_rtrip_time += now() - lm_rtrip_time
                    nb_lm_inferences += 1
                    self.lm_postprocess(r, inference)
                    self.lm_render(annotated_frame, r)

            if not self.crop:
                annotated_frame = annotated_frame[pad_h : pad_h + h, pad_w : pad_w + w]

            self.fps.display(annotated_frame, orig=(50, 50), color=(240, 180, 100))
            cv2.imshow("video", annotated_frame)

            key = cv2.waitKey(1)
            if key == ord("q") or key == 27:
                break
            elif key == 32:
                # Pause on space bar
                cv2.waitKey(0)
            elif key == ord("1"):
                self.show_pd_box = not self.show_pd_box
            elif key == ord("2"):
                self.show_pd_kps = not self.show_pd_kps
            elif key == ord("3"):
                self.show_rot_rect = not self.show_rot_rect
            elif key == ord("4"):
                self.show_landmarks = not self.show_landmarks
            elif key == ord("5"):
                self.show_handedness = not self.show_handedness
            elif key == ord("6"):
                self.show_scores = not self.show_scores
            elif key == ord("7"):
                self.show_gesture = not self.show_gesture

        # Print some stats
        print(f"# palm detection inferences : {nb_pd_inferences}")
        print(f"# hand landmark inferences  : {nb_lm_inferences}")
        print(f"Palm detection round trip   : {glob_pd_rtrip_time/nb_pd_inferences*1000:.1f} ms")
        print(f"Hand landmark round trip    : {glob_lm_rtrip_time/nb_lm_inferences*1000:.1f} ms")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-i", "--input", type=str, default="0", help="Path to video or image file to use as input (default=%(default)s)"
#     )
#     parser.add_argument("-g", "--gesture", action="store_true", help="enable gesture recognition")
#     parser.add_argument(
#         "--pd_m",
#         default="models/palm_detection_FP32.xml",
#         type=str,
#         help="Path to an .xml file for palm detection model (default=%(default)s)",
#     )
#     parser.add_argument(
#         "--pd_device", default="CPU", type=str, help="Target device for the palm detection model (default=%(default)s)"
#     )
#     parser.add_argument(
#         "--no_lm", action="store_true", help="only the palm detection model is run, not the hand landmark model"
#     )
#     parser.add_argument(
#         "--lm_m",
#         default="models/hand_landmark_FP32.xml",
#         type=str,
#         help="Path to an .xml file for landmark model (default=%(default)s)",
#     )
#     parser.add_argument(
#         "--lm_device",
#         default="CPU",
#         type=str,
#         help="Target device for the landmark regression model (default=%(default)s)",
#     )
#     parser.add_argument(
#         "-c",
#         "--crop",
#         action="store_true",
#         help="center crop frames to a square shape before feeding palm detection model",
#     )
#     parser.add_argument(
#         "-g",
#         "--get",
#         action="store_false",
#         help="get data for training gesture recognition",
#     )

#     args = parser.parse_args()

# ht = HandTracker(
#     input_src=args.input,
#     pd_device=args.pd_device,
#     use_lm=not args.no_lm,
#     lm_device=args.lm_device,
#     use_gesture=args.gesture,
#     crop=args.crop,
# )
# ht.run()
