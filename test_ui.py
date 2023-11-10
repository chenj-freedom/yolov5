import sys
import argparse
import csv
import os
import platform
from pathlib import Path
import numpy as np
import time

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
from PyQt5.QtGui import QImage, QPixmap

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

class ObjectDetectionUI(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()
        self.init_data()

    def init_ui(self):
        self.setWindowTitle('Object Detection UI')
        self.setGeometry(100, 100, 800, 600)

        # 创建视频预览的标签
        self.video_label = QLabel(self)
        self.video_label.setText('Video Preview')
        self.video_label.setAlignment(Qt.AlignCenter)

        # 创建上传视频按钮
        self.upload_button = QPushButton('上传视频', self)
        self.upload_button.clicked.connect(self.upload_video)

        # 创建开始检测按钮
        self.detect_button = QPushButton('开始检测', self)
        self.detect_button.clicked.connect(self.start_detection)

        # 布局管理
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.detect_button)

        self.setLayout(layout)

    def init_data(self):
        self.show_black_frame = True
        self.count = 0

        self.load_model()

    def upload_video(self):
        # 打开文件对话框选择视频文件
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setNameFilter('Video Files (*.mp4 *.avi)')
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setViewMode(QFileDialog.Detail)
        file_path, _ = file_dialog.getOpenFileName(self, '选择视频文件', '', 'Video Files (*.mp4 *.avi)', options=options)
        self.source = file_path

        # 显示预览
        if file_path:
            # 使用OpenCV读取视频
            cap = cv2.VideoCapture(file_path)

            # 读取第一帧
            ret, frame = cap.read()
            height, width, _ = frame.shape

            # 定义目标显示大小
            target_width = width
            target_height = height

            # 缩放图像以适应目标大小
            scaled_frame = cv2.resize(frame, (target_width, target_height))

            # 将OpenCV的BGR格式转换为RGB格式
            rgb_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)

            # 将RGB图像转换为Qt可用的QImage
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 将QImage显示在label中
            self.video_label.setPixmap(QPixmap.fromImage(q_image))

            # 释放OpenCV视频捕获对象
            cap.release()

    def load_model(self,
                   weights=ROOT / 'pretrained/best.pt',  # model path or triton URL
                   source=ROOT / 'data/videos/input.mp4',  # file/dir/URL/glob/screen/0(webcam)
                   data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
                   imgsz=(1280, 720),  # inference size (height, width)
                   device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                   half=False,  # use FP16 half-precision inference
                   dnn=False,  # use OpenCV DNN for ONNX inference
                   ):
        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

    def start_detection(self):
        self.run()

    def update_frame(self):
        print("into udpate_frame..show_black_frame=" + str(self.show_black_frame))
        if self.show_black_frame:
            image = QImage(1280, 720, QImage.Format_RGB32)
            image.fill(QColor(0, 0, 0))
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(pixmap)
        else:
            image = QImage(1280, 720, QImage.Format_RGB32)
            image.fill(QColor(255, 255, 255))
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(pixmap)

        self.show_black_frame = not self.show_black_frame

        QApplication.processEvents()

    def run(self,
            source=ROOT / 'data/videos/input.mp4',  # file/dir/URL/glob/screen/0(webcam)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            vid_stride=1,  # video frame-rate stride
    ):
        source = str(self.source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Dataloader
        bs = 1  # batch_size
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                pred = self.model(im, augment=augment, visualize=False)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = self.names[c] if hide_conf else f'{self.names[c]}'
                        confidence = float(conf)
                        confidence_str = f'{confidence:.2f}'

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))

                # Stream results
                im0 = annotator.result()
                if view_img:
                    # cv2.imshow(str(p), im0)

                    height, width, channel = im0.shape
                    bytesPerLine = 3 * width
                    q_image = QImage(im0.data, width, height, bytesPerLine, QImage.Format_BGR888)
                    self.video_label.setPixmap(QPixmap.fromImage(q_image))

                    QApplication.processEvents()    # force refresh

                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = ObjectDetectionUI()
    ui.show()
    sys.exit(app.exec_())
