import sys
import os
import json
import time
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
import requests
import io
import urllib.request
import zipfile
import tempfile
from tqdm import tqdm
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QComboBox, QProgressBar, QScrollArea, QGridLayout,
                            QMessageBox, QTabWidget, QSplitter, QTextEdit,
                            QInputDialog, QDialog, QLineEdit, QFormLayout,
                            QListWidget, QListWidgetItem, QMenu,
                            QColorDialog, QSpinBox, QDoubleSpinBox, QGroupBox)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QCursor, QAction
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal, QThread, QUrl
class ImageAnnotation:
    def __init__(self, image_path):
        self.image_path = image_path
        self.boxes = []  
        self.class_names = []
        self.image_id = None  
        self.image_width = None
        self.image_height = None
        self._load_image_dimensions()
    def _load_image_dimensions(self):
        try:
            with Image.open(self.image_path) as img:
                self.image_width, self.image_height = img.size
        except Exception as e:
            print(f"Ошибка при загрузке размеров изображения: {e}")
    def add_box(self, x, y, width, height, class_id=0):
        self.boxes.append([x, y, width, height, class_id])
    def remove_box(self, index):
        if 0 <= index < len(self.boxes):
            self.boxes.pop(index)
    def to_json(self):
        return {
            'image_path': self.image_path,
            'boxes': self.boxes,
            'class_names': self.class_names,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'image_id': self.image_id
        }
    @classmethod
    def from_json(cls, data):
        annotation = cls(data['image_path'])
        annotation.boxes = data['boxes']
        annotation.class_names = data['class_names']
        annotation.image_width = data.get('image_width')
        annotation.image_height = data.get('image_height')
        annotation.image_id = data.get('image_id')
        return annotation
    @classmethod
    def from_yolo_file(cls, image_path, label_path):
        annotation = cls(image_path)
        if not annotation.image_width or not annotation.image_height:
            annotation._load_image_dimensions()
        if not annotation.image_width or not annotation.image_height:
            return annotation
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(float(parts[0]))
                        x_center = float(parts[1]) * annotation.image_width
                        y_center = float(parts[2]) * annotation.image_height
                        width = float(parts[3]) * annotation.image_width
                        height = float(parts[4]) * annotation.image_height
                        x = int(x_center - width / 2)
                        y = int(y_center - height / 2)
                        annotation.add_box(x, y, int(width), int(height), class_id)
        except Exception as e:
            print(f"Ошибка при загрузке файла YOLO: {e}")
        return annotation
    @classmethod
    def from_coco_annotation(cls, image_path, annotation_data, image_id):
        annotation = cls(image_path)
        boxes = []
        for ann in annotation_data.get('annotations', []):
            if ann.get('image_id') == image_id:
                bbox = ann.get('bbox', [])
                if len(bbox) == 4:
                    x, y, w, h = [int(coord) for coord in bbox]
                    class_id = ann.get('category_id', 0)
                    boxes.append([x, y, w, h, class_id])
        annotation.boxes = boxes
        for img in annotation_data.get('images', []):
            if img.get('id') == image_id:
                annotation.image_width = img.get('width')
                annotation.image_height = img.get('height')
                annotation.image_id = image_id
                break
        return annotation
    def to_yolo_format(self):
        if not self.image_width or not self.image_height:
            self._load_image_dimensions()
        yolo_annotations = []
        for box in self.boxes:
            x, y, w, h, class_id = box
            x_center = (x + w / 2) / self.image_width
            y_center = (y + h / 2) / self.image_height
            norm_width = w / self.image_width
            norm_height = h / self.image_height
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
        return yolo_annotations
    def to_coco_format(self, image_id, annotation_id_start=0):
        self.image_id = image_id
        if not self.image_width or not self.image_height:
            self._load_image_dimensions()
        image_info = {
            "id": image_id,
            "file_name": os.path.basename(self.image_path),
            "width": self.image_width,
            "height": self.image_height
        }
        annotations = []
        for i, box in enumerate(self.boxes):
            x, y, w, h, class_id = box
            annotation = {
                "id": annotation_id_start + i,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            }
            annotations.append(annotation)
        return image_info, annotations
class ObjectDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, class_count=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.class_count = class_count
        self.image_files = [f for f in os.listdir(images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + ".txt")
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) == 5:
                        class_id, x_center, y_center, width, height = map(float, data)
                        x_min = (x_center - width / 2) * img_width
                        y_min = (y_center - height / 2) * img_height
                        x_max = (x_center + width / 2) * img_width
                        y_max = (y_center + height / 2) * img_height
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(int(class_id))
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        if self.transform:
            image = self.transform(image)
        return image, target
class ModelDownloader:
    AVAILABLE_MODELS = {
        "mobilenet_v2": {
            "url": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
            "description": "MobileNetV2 - легкая модель для мобильных устройств",
            "type": "classification"
        },
        "resnet50": {
            "url": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
            "description": "ResNet50 - глубокая сверточная сеть с остаточными связями",
            "type": "classification"
        },
        "yolov5s": {
            "url": "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt",
            "description": "YOLOv5 Small - компактная модель для обнаружения объектов",
            "type": "detection"
        },
        "ssd_mobilenet": {
            "url": "https://storage.googleapis.com/models-zoo/object_detection/ssd_mobilenet.pt",
            "description": "SSD MobileNet - модель для обнаружения объектов на основе MobileNet",
            "type": "detection"
        }
    }
    @staticmethod
    def get_available_models():
        return ModelDownloader.AVAILABLE_MODELS
    @staticmethod
    def download_model(model_name, target_dir, progress_callback=None):
        if model_name not in ModelDownloader.AVAILABLE_MODELS:
            raise ValueError(f"Модель {model_name} не найдена в списке доступных моделей")
        model_info = ModelDownloader.AVAILABLE_MODELS[model_name]
        url = model_info["url"]
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, f"{model_name}.pt")
        try:
            response = requests.head(url, allow_redirects=True)
            file_size = int(response.headers.get('content-length', 0))
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(target_path, 'wb') as f:
                    chunk_size = 8192
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:  
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress = int(downloaded * 100 / file_size) if file_size > 0 else 0
                            if progress_callback:
                                progress_callback(progress)
            try:
                model_data = torch.load(target_path, map_location=torch.device('cpu'))
                if isinstance(model_data, dict) and "model_type" not in model_data:
                    model_metadata = {
                        "model_type": model_name,
                        "source": "pretrained",
                        "url": url,
                        "description": model_info["description"],
                        "type": model_info["type"],
                        "model_state_dict": model_data,
                        "transform_params": {
                            "mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]
                        }
                    }
                    torch.save(model_metadata, target_path)
                return target_path
            except Exception as e:
                os.remove(target_path)
                raise ValueError(f"Загруженный файл не является корректной моделью PyTorch: {str(e)}")
        except Exception as e:
            raise ValueError(f"Ошибка при загрузке модели: {str(e)}")
class ClassManager(QDialog):
    def __init__(self, class_names=None, class_colors=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Управление классами объектов")
        self.resize(500, 400)
        self.class_names = class_names or []
        self.class_colors = class_colors or []
        if not self.class_colors and self.class_names:
            default_colors = [
                QColor(255, 0, 0),    
                QColor(0, 255, 0),    
                QColor(0, 0, 255),    
                QColor(255, 255, 0),  
                QColor(255, 165, 0),  
                QColor(128, 0, 128)   
            ]
            for i in range(len(self.class_names)):
                self.class_colors.append(default_colors[i % len(default_colors)])
        self.init_ui()
        self.update_class_list()
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.class_list = QListWidget()
        self.class_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.class_list.itemClicked.connect(self.on_class_selected)
        layout.addWidget(self.class_list)
        form_layout = QFormLayout()
        self.class_id_edit = QLineEdit()
        self.class_id_edit.setReadOnly(True)  
        form_layout.addRow("ID класса:", self.class_id_edit)
        self.class_name_edit = QLineEdit()
        form_layout.addRow("Название класса:", self.class_name_edit)
        color_layout = QHBoxLayout()
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(30, 30)
        self.color_preview.setStyleSheet("background-color: red; border: 1px solid black;")
        color_layout.addWidget(self.color_preview)
        self.color_button = QPushButton("Выбрать цвет")
        self.color_button.clicked.connect(self.choose_color)
        color_layout.addWidget(self.color_button)
        form_layout.addRow("Цвет класса:", color_layout)
        layout.addLayout(form_layout)
        buttons_layout = QHBoxLayout()
        self.add_button = QPushButton("Добавить")
        self.add_button.clicked.connect(self.add_class)
        buttons_layout.addWidget(self.add_button)
        self.update_button = QPushButton("Обновить")
        self.update_button.clicked.connect(self.update_class)
        self.update_button.setEnabled(False)  
        buttons_layout.addWidget(self.update_button)
        self.delete_button = QPushButton("Удалить")
        self.delete_button.clicked.connect(self.delete_class)
        self.delete_button.setEnabled(False)  
        buttons_layout.addWidget(self.delete_button)
        layout.addLayout(buttons_layout)
        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
    def update_class_list(self):
        self.class_list.clear()
        for i, class_name in enumerate(self.class_names):
            item = QListWidgetItem(f"{i}: {class_name}")
            item.setData(Qt.ItemDataRole.UserRole, i)
            if i < len(self.class_colors):
                color = self.class_colors[i]
                item.setBackground(color)
                brightness = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
                if brightness < 128:
                    item.setForeground(Qt.GlobalColor.white)
            self.class_list.addItem(item)
    def on_class_selected(self, item):
        class_id = item.data(Qt.ItemDataRole.UserRole)
        class_name = self.class_names[class_id]
        self.class_id_edit.setText(str(class_id))
        self.class_name_edit.setText(class_name)
        if class_id < len(self.class_colors):
            color = self.class_colors[class_id]
            self.color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")
        self.update_button.setEnabled(True)
        self.delete_button.setEnabled(True)
    def add_class(self):
        class_name = self.class_name_edit.text().strip()
        if not class_name:
            QMessageBox.warning(self, "Ошибка", "Введите название класса")
            return
        color_str = self.color_preview.styleSheet().split("background-color:")[1].split(";")[0].strip()
        color = QColor(color_str)
        self.class_names.append(class_name)
        self.class_colors.append(color)
        self.update_class_list()
        self.class_id_edit.clear()
        self.class_name_edit.clear()
    def update_class(self):
        if not self.class_list.currentItem():
            return
        class_id = int(self.class_id_edit.text())
        class_name = self.class_name_edit.text().strip()
        if not class_name:
            QMessageBox.warning(self, "Ошибка", "Введите название класса")
            return
        color_str = self.color_preview.styleSheet().split("background-color:")[1].split(";")[0].strip()
        color = QColor(color_str)
        self.class_names[class_id] = class_name
        if class_id < len(self.class_colors):
            self.class_colors[class_id] = color
        else:
            self.class_colors.append(color)
        self.update_class_list()
    def delete_class(self):
        if not self.class_list.currentItem():
            return
        class_id = int(self.class_id_edit.text())
        reply = QMessageBox.question(
            self, 
            "Подтверждение", 
            f"Вы уверены, что хотите удалить класс {class_id}: {self.class_names[class_id]}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.class_names.pop(class_id)
            if class_id < len(self.class_colors):
                self.class_colors.pop(class_id)
            self.update_class_list()
            self.class_id_edit.clear()
            self.class_name_edit.clear()
            self.update_button.setEnabled(False)
            self.delete_button.setEnabled(False)
    def choose_color(self):
        if not self.class_list.currentItem():
            return
        class_id = int(self.class_id_edit.text())
        color = QColorDialog.getColor(self.class_colors[class_id] if class_id < len(self.class_colors) else Qt.GlobalColor.red, self)
        if color.isValid():
            self.color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")
            if class_id < len(self.class_colors):
                self.class_colors[class_id] = color
            else:
                self.class_colors.append(color)
class SimpleObjectDetector(nn.Module):
    def __init__(self, num_classes):
        super(SimpleObjectDetector, self).__init__()
        self.backbone = models.mobilenet_v2(pretrained=True).features
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.conv = nn.Conv2d(1280, 256, kernel_size=1)
        self.box_predictor = nn.Conv2d(256, 4, kernel_size=1)  
        self.class_predictor = nn.Conv2d(256, num_classes, kernel_size=1)
    def forward(self, x):
        features = self.backbone(x)
        features = self.conv(features)
        boxes = self.box_predictor(features)
        classes = self.class_predictor(features)
        boxes = boxes.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        classes = classes.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, classes.size(1))
        return {'boxes': boxes, 'scores': torch.sigmoid(classes)}
class TrainingThread(QThread):
    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    training_finished = pyqtSignal(str)
    def __init__(self, model_type, dataset_path, epochs=10, batch_size=4, learning_rate=0.001):
        super().__init__()
        self.model_type = model_type
        self.dataset_path = dataset_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_running = True
        self.model_save_path = os.path.join(os.path.dirname(dataset_path), "models", 
                                           f"{model_type.lower()}_{time.strftime('%Y%m%d_%H%M%S')}.pt")
    def run(self):
        try:
            self.log_updated.emit(f"Начинаем обучение модели {self.model_type}...")
            transform = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            classes_file = os.path.join(self.dataset_path, "classes.txt")
            class_names = []
            if os.path.exists(classes_file):
                with open(classes_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(':')
                        if len(parts) >= 2:
                            class_names.append(parts[1].strip())
            num_classes = len(class_names) if class_names else 1
            self.log_updated.emit(f"Найдено классов: {num_classes}")
            train_images_dir = os.path.join(self.dataset_path, "train", "images")
            train_labels_dir = os.path.join(self.dataset_path, "train", "labels")
            val_images_dir = os.path.join(self.dataset_path, "val", "images")
            val_labels_dir = os.path.join(self.dataset_path, "val", "labels")
            train_dataset = ObjectDetectionDataset(train_images_dir, train_labels_dir, transform, num_classes)
            val_dataset = ObjectDetectionDataset(val_images_dir, val_labels_dir, transform, num_classes)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            self.log_updated.emit(f"Размер обучающего набора: {len(train_dataset)}")
            self.log_updated.emit(f"Размер валидационного набора: {len(val_dataset)}")
            if self.model_type == "MobileNet":
                model = SimpleObjectDetector(num_classes + 1)  
            elif self.model_type == "YOLO":
                model = SimpleObjectDetector(num_classes + 1)
                self.log_updated.emit("Используется упрощенная модель вместо YOLO")
            else:
                model = SimpleObjectDetector(num_classes + 1)
                self.log_updated.emit("Используется упрощенная модель")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            self.log_updated.emit(f"Используется устройство: {device}")
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            for epoch in range(self.epochs):
                if not self.is_running:
                    break
                model.train()
                running_loss = 0.0
                for i, (images, targets) in enumerate(train_loader):
                    if not self.is_running:
                        break
                    images = images.to(device)
                    target_boxes = [t['boxes'].to(device) for t in targets]
                    target_labels = [t['labels'].to(device) for t in targets]
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = torch.tensor(0.0, device=device)
                    for j in range(len(images)):
                        if len(target_boxes[j]) > 0:
                            pred_boxes = outputs['boxes'][j][:len(target_boxes[j])]
                            loss += torch.abs(pred_boxes - target_boxes[j]).mean()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    progress = (epoch * len(train_loader) + i) * 100 // (self.epochs * len(train_loader))
                    self.progress_updated.emit(progress)
                    if i % 5 == 0:
                        self.log_updated.emit(f"Эпоха {epoch+1}/{self.epochs}, шаг {i+1}/{len(train_loader)}, потери: {loss.item():.4f}")
                epoch_loss = running_loss / len(train_loader)
                self.log_updated.emit(f"Эпоха {epoch+1}/{self.epochs} завершена, средние потери: {epoch_loss:.4f}")
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            model_info = {
                "model_state_dict": model.state_dict(),
                "model_type": self.model_type,
                "num_classes": num_classes,
                "class_names": class_names,
                "transform_params": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                }
            }
            torch.save(model_info, self.model_save_path)
            self.log_updated.emit(f"Модель сохранена в {self.model_save_path}")
            self.training_finished.emit(self.model_save_path)
        except Exception as e:
            import traceback
            self.log_updated.emit(f"Ошибка при обучении: {str(e)}")
            self.log_updated.emit(traceback.format_exc())
    def stop(self):
        self.is_running = False
class ImageWidget(QLabel):
    box_added = pyqtSignal(int, int, int, int)  
    box_selected = pyqtSignal(int)  
    box_edited = pyqtSignal(int, int, int, int, int)  
    box_deleted = pyqtSignal(int)  
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 1px solid gray;")
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.current_rect = QRect()
        self.boxes = []  
        self.colors = [QColor(255, 0, 0), QColor(0, 255, 0), 
                      QColor(0, 0, 255), QColor(255, 255, 0),
                      QColor(255, 165, 0), QColor(128, 0, 128)]
        self.scale_factor = 1.0
        self.original_pixmap = None
        self.selected_box_index = -1  
        self.editing_box = False
        self.drag_start_point = QPoint()
        self.drag_box_initial = None  
        self.resize_handle = None  
        self.resize_threshold = 10  
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
    def set_image(self, image_path):
        if not os.path.exists(image_path):
            return False
        self.original_pixmap = QPixmap(image_path)
        if self.original_pixmap.isNull():
            return False
        self.selected_box_index = -1
        self.update_display()
        return True
    def update_display(self):
        if self.original_pixmap is None:
            return
        scaled_pixmap = self.original_pixmap.scaled(
            self.width() * self.scale_factor, 
            self.height() * self.scale_factor,
            Qt.AspectRatioMode.KeepAspectRatio
        )
        self.setPixmap(scaled_pixmap)
    def set_boxes(self, boxes):
        self.boxes = boxes
        self.selected_box_index = -1  
        self.update()
    def set_colors(self, colors):
        self.colors = colors
        self.update()
    def get_box_rect(self, box_index):
        if 0 <= box_index < len(self.boxes):
            box = self.boxes[box_index]
            x, y, w, h, _ = box
            rel_x = x / self.original_pixmap.width()
            rel_y = y / self.original_pixmap.height()
            rel_w = w / self.original_pixmap.width()
            rel_h = h / self.original_pixmap.height()
            pixmap_rect = self.get_pixmap_rect()
            rect_x = int(pixmap_rect.x() + rel_x * pixmap_rect.width())
            rect_y = int(pixmap_rect.y() + rel_y * pixmap_rect.height())
            rect_w = int(rel_w * pixmap_rect.width())
            rect_h = int(rel_h * pixmap_rect.height())
            return QRect(rect_x, rect_y, rect_w, rect_h)
        return QRect()
    def get_box_at_position(self, pos):
        for i in range(len(self.boxes) - 1, -1, -1):  
            rect = self.get_box_rect(i)
            if rect.contains(pos):
                return i
        return -1
    def get_resize_handle(self, pos, box_index):
        if box_index < 0:
            return None
        rect = self.get_box_rect(box_index)
        threshold = self.resize_threshold
        if QRect(rect.topLeft().x() - threshold, rect.topLeft().y() - threshold, 
                threshold * 2, threshold * 2).contains(pos):
            return "top-left"
        elif QRect(rect.topRight().x() - threshold, rect.topRight().y() - threshold, 
                 threshold * 2, threshold * 2).contains(pos):
            return "top-right"
        elif QRect(rect.bottomLeft().x() - threshold, rect.bottomLeft().y() - threshold, 
                 threshold * 2, threshold * 2).contains(pos):
            return "bottom-left"
        elif QRect(rect.bottomRight().x() - threshold, rect.bottomRight().y() - threshold, 
                 threshold * 2, threshold * 2).contains(pos):
            return "bottom-right"
        return None
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.original_pixmap:
            pos = event.position().toPoint()
            box_index = self.get_box_at_position(pos)
            if box_index >= 0:
                self.selected_box_index = box_index
                self.box_selected.emit(box_index)
                self.resize_handle = self.get_resize_handle(pos, box_index)
                if self.resize_handle:
                    self.editing_box = True
                    self.drag_start_point = pos
                    self.drag_box_initial = self.get_box_rect(box_index)
                else:
                    self.editing_box = True
                    self.drag_start_point = pos
                    self.drag_box_initial = self.get_box_rect(box_index)
            else:
                self.drawing = True
                self.start_point = pos
                self.end_point = self.start_point
                self.current_rect = QRect()
                self.selected_box_index = -1  
            self.update()
    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self.drawing:
            self.end_point = pos
            self.current_rect = QRect(self.start_point, self.end_point).normalized()
            self.update()
        elif self.editing_box and self.selected_box_index >= 0:
            if self.resize_handle:
                rect = self.drag_box_initial
                if self.resize_handle == "top-left":
                    new_rect = QRect(
                        rect.left() + (pos.x() - self.drag_start_point.x()),
                        rect.top() + (pos.y() - self.drag_start_point.y()),
                        rect.width() - (pos.x() - self.drag_start_point.x()),
                        rect.height() - (pos.y() - self.drag_start_point.y())
                    )
                elif self.resize_handle == "top-right":
                    new_rect = QRect(
                        rect.left(),
                        rect.top() + (pos.y() - self.drag_start_point.y()),
                        rect.width() + (pos.x() - self.drag_start_point.x()),
                        rect.height() - (pos.y() - self.drag_start_point.y())
                    )
                elif self.resize_handle == "bottom-left":
                    new_rect = QRect(
                        rect.left() + (pos.x() - self.drag_start_point.x()),
                        rect.top(),
                        rect.width() - (pos.x() - self.drag_start_point.x()),
                        rect.height() + (pos.y() - self.drag_start_point.y())
                    )
                elif self.resize_handle == "bottom-right":
                    new_rect = QRect(
                        rect.left(),
                        rect.top(),
                        rect.width() + (pos.x() - self.drag_start_point.x()),
                        rect.height() + (pos.y() - self.drag_start_point.y())
                    )
                self.current_rect = new_rect.normalized()
            else:
                dx = pos.x() - self.drag_start_point.x()
                dy = pos.y() - self.drag_start_point.y()
                rect = self.drag_box_initial
                new_rect = QRect(
                    rect.left() + dx,
                    rect.top() + dy,
                    rect.width(),
                    rect.height()
                )
                self.current_rect = new_rect
            self.update()
        else:
            box_index = self.get_box_at_position(pos)
            if box_index >= 0:
                resize_handle = self.get_resize_handle(pos, box_index)
                if resize_handle:
                    if resize_handle in ["top-left", "bottom-right"]:
                        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                    else:
                        self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                else:
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.drawing:
                self.drawing = False
                self.end_point = event.position().toPoint()
                rect = QRect(self.start_point, self.end_point).normalized()
                if rect.width() > 10 and rect.height() > 10:
                    pixmap_rect = self.get_pixmap_rect()
                    if not pixmap_rect.isNull():
                        x = (rect.x() - pixmap_rect.x()) / pixmap_rect.width()
                        y = (rect.y() - pixmap_rect.y()) / pixmap_rect.height()
                        w = rect.width() / pixmap_rect.width()
                        h = rect.height() / pixmap_rect.height()
                        if 0 <= x <= 1 and 0 <= y <= 1:
                            img_width = self.original_pixmap.width()
                            img_height = self.original_pixmap.height()
                            abs_x = int(x * img_width)
                            abs_y = int(y * img_height)
                            abs_w = int(w * img_width)
                            abs_h = int(h * img_height)
                            self.box_added.emit(abs_x, abs_y, abs_w, abs_h)
            elif self.editing_box and self.selected_box_index >= 0:
                self.editing_box = False
                pixmap_rect = self.get_pixmap_rect()
                if not pixmap_rect.isNull() and not self.current_rect.isNull():
                    x = (self.current_rect.x() - pixmap_rect.x()) / pixmap_rect.width()
                    y = (self.current_rect.y() - pixmap_rect.y()) / pixmap_rect.height()
                    w = self.current_rect.width() / pixmap_rect.width()
                    h = self.current_rect.height() / pixmap_rect.height()
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        img_width = self.original_pixmap.width()
                        img_height = self.original_pixmap.height()
                        abs_x = max(0, int(x * img_width))
                        abs_y = max(0, int(y * img_height))
                        abs_w = min(img_width - abs_x, int(w * img_width))
                        abs_h = min(img_height - abs_y, int(h * img_height))
                        if abs_w > 5 and abs_h > 5:
                            self.box_edited.emit(self.selected_box_index, abs_x, abs_y, abs_w, abs_h)
                self.current_rect = QRect()
            self.update()
    def get_pixmap_rect(self):
        if not self.pixmap() or self.pixmap().isNull():
            return QRect()
        widget_size = self.size()
        pixmap_size = self.pixmap().size()
        scaled_width = min(widget_size.width(), pixmap_size.width())
        scaled_height = min(widget_size.height(), pixmap_size.height())
        x = (widget_size.width() - scaled_width) // 2
        y = (widget_size.height() - scaled_height) // 2
        return QRect(x, y, scaled_width, scaled_height)
    def show_context_menu(self, position):
        if not self.original_pixmap:
            return
        box_index = self.get_box_at_position(position)
        if box_index >= 0:
            context_menu = QMenu(self)
            delete_action = QAction("Удалить выделение", self)
            delete_action.triggered.connect(lambda: self.box_deleted.emit(box_index))
            context_menu.addAction(delete_action)
            context_menu.exec(QCursor.pos())
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.pixmap() and not self.pixmap().isNull():
            painter = QPainter(self)
            pixmap_rect = self.get_pixmap_rect()
            if not pixmap_rect.isNull() and self.original_pixmap:
                for i, box in enumerate(self.boxes):
                    x, y, w, h, class_id = box
                    rel_x = x / self.original_pixmap.width()
                    rel_y = y / self.original_pixmap.height()
                    rel_w = w / self.original_pixmap.width()
                    rel_h = h / self.original_pixmap.height()
                    rect_x = int(pixmap_rect.x() + rel_x * pixmap_rect.width())
                    rect_y = int(pixmap_rect.y() + rel_y * pixmap_rect.height())
                    rect_w = int(rel_w * pixmap_rect.width())
                    rect_h = int(rel_h * pixmap_rect.height())
                    color = self.colors[class_id % len(self.colors)]
                    if i == self.selected_box_index:
                        painter.setPen(QPen(color, 3, Qt.PenStyle.DashLine))
                    else:
                        painter.setPen(QPen(color, 2, Qt.PenStyle.SolidLine))
                    painter.drawRect(rect_x, rect_y, rect_w, rect_h)
                    painter.setPen(QPen(Qt.GlobalColor.white))
                    painter.fillRect(rect_x, rect_y - 20, 50, 20, color)
                    painter.drawText(rect_x + 5, rect_y - 5, f"Класс {class_id}")
            if self.drawing and not self.current_rect.isNull():
                painter.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine))
                painter.drawRect(self.current_rect)
            elif self.editing_box and not self.current_rect.isNull():
                painter.setPen(QPen(Qt.GlobalColor.green, 2, Qt.PenStyle.DashLine))
                painter.drawRect(self.current_rect)
                if self.selected_box_index >= 0:
                    painter.setBrush(QColor(0, 255, 0))
                    painter.setPen(QPen(Qt.GlobalColor.black, 1))
                    marker_size = 8
                    painter.drawRect(self.current_rect.topLeft().x() - marker_size // 2, 
                                    self.current_rect.topLeft().y() - marker_size // 2, 
                                    marker_size, marker_size)
                    painter.drawRect(self.current_rect.topRight().x() - marker_size // 2, 
                                    self.current_rect.topRight().y() - marker_size // 2, 
                                    marker_size, marker_size)
                    painter.drawRect(self.current_rect.bottomLeft().x() - marker_size // 2, 
                                    self.current_rect.bottomLeft().y() - marker_size // 2, 
                                    marker_size, marker_size)
                    painter.drawRect(self.current_rect.bottomRight().x() - marker_size // 2, 
                                    self.current_rect.bottomRight().y() - marker_size // 2, 
                                    marker_size, marker_size)
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroMark")
        self.setMinimumSize(1000, 700)
        self.annotations = {}  
        self.current_image_path = None
        self.current_class_id = 0
        self.model_path = None
        self.project_dir = None
        self.class_colors = []  
        self.default_colors = [
            QColor(255, 0, 0),    
            QColor(0, 255, 0),    
            QColor(0, 0, 255),    
            QColor(255, 255, 0),  
            QColor(255, 165, 0),  
            QColor(128, 0, 128)   
        ]
        self.init_ui()
        self.set_app_icon()
    def init_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        self.create_menu()
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.create_annotation_tab(), "Разметка")
        self.tab_widget.addTab(self.create_training_tab(), "Обучение")
        self.tab_widget.addTab(self.create_testing_tab(), "Тестирование")
        main_layout.addWidget(self.tab_widget)
        toolbar_layout = QHBoxLayout()
        self.btn_new_project = QPushButton("Новый проект")
        self.btn_new_project.clicked.connect(self.create_new_project)
        self.btn_new_project.setToolTip("Создание нового проекта")
        toolbar_layout.addWidget(self.btn_new_project)
        self.btn_save_project = QPushButton("Сохранить проект")
        self.btn_save_project.clicked.connect(self.save_project)
        self.btn_save_project.setToolTip("Сохранение текущего проекта")
        toolbar_layout.addWidget(self.btn_save_project)
        self.btn_load_project = QPushButton("Загрузить проект")
        self.btn_load_project.clicked.connect(self.load_project)
        self.btn_load_project.setToolTip("Загрузка существующего проекта")
        toolbar_layout.addWidget(self.btn_load_project)
        self.btn_import = QPushButton("Импорт разметок")
        self.btn_import.clicked.connect(self.show_import_dialog)
        self.btn_import.setToolTip("Импорт существующих разметок в форматах YOLO, COCO или JSON")
        toolbar_layout.addWidget(self.btn_import)
        main_layout.addLayout(toolbar_layout)
        self.setCentralWidget(central_widget)      
    def set_app_icon(self):
        icon_path = "NeuroMark.png"
        if os.path.exists(icon_path):
            from PyQt6.QtGui import QIcon
            self.setWindowIcon(QIcon(icon_path))
    def create_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Файл")
        new_project_action = QAction("Новый проект", self)
        new_project_action.triggered.connect(self.create_new_project)
        file_menu.addAction(new_project_action)
        save_project_action = QAction("Сохранить проект", self)
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)
        load_project_action = QAction("Загрузить проект", self)
        load_project_action.triggered.connect(self.load_project)
        file_menu.addAction(load_project_action)
        file_menu.addSeparator()
        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        import_menu = menu_bar.addMenu("Импорт")
        import_yolo_action = QAction("Импорт разметок YOLO", self)
        import_yolo_action.triggered.connect(lambda: self.show_import_dialog(0))
        import_menu.addAction(import_yolo_action)
        import_coco_action = QAction("Импорт разметок COCO", self)
        import_coco_action.triggered.connect(lambda: self.show_import_dialog(1))
        import_menu.addAction(import_coco_action)
        import_json_action = QAction("Импорт разметок JSON", self)
        import_json_action.triggered.connect(lambda: self.show_import_dialog(2))
        import_menu.addAction(import_json_action)
    def show_import_dialog(self, tab_index=0):
        import_dialog = ImportDialog(self)
        if hasattr(import_dialog, 'tab_widget') and 0 <= tab_index <= 2:
            import_dialog.tab_widget.setCurrentIndex(tab_index)
        import_dialog.exec()
    def create_annotation_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.image_widget = ImageWidget()
        self.image_widget.box_added.connect(self.on_box_added)
        self.image_widget.box_selected.connect(self.on_box_selected)
        self.image_widget.box_edited.connect(self.on_box_edited)
        self.image_widget.box_deleted.connect(self.on_box_deleted)
        self.image_widget.setToolTip("Здесь отображается изображение. Нажмите и перетащите для создания разметки.")
        left_layout.addWidget(self.image_widget)
        tools_layout = QHBoxLayout()
        class_label = QLabel("Класс:")
        tools_layout.addWidget(class_label)
        self.class_combo = QComboBox()
        self.class_combo.addItem("Класс 0")
        self.class_combo.currentIndexChanged.connect(self.on_class_changed)
        self.class_combo.setToolTip("Выберите класс объекта для разметки")
        tools_layout.addWidget(self.class_combo)
        btn_manage_classes = QPushButton("Управление классами")
        btn_manage_classes.clicked.connect(self.add_class)
        btn_manage_classes.setToolTip("Добавление, редактирование и удаление классов объектов")
        tools_layout.addWidget(btn_manage_classes)
        btn_remove_box = QPushButton("Удалить выделение")
        btn_remove_box.clicked.connect(self.remove_selected_box)
        btn_remove_box.setToolTip("Удаление выбранного объекта")
        tools_layout.addWidget(btn_remove_box)
        left_layout.addLayout(tools_layout)
        self.selected_box_info = QLabel("Выберите объект для отображения информации")
        left_layout.addWidget(self.selected_box_info)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        image_buttons_layout = QHBoxLayout()
        btn_load_images = QPushButton("Загрузить изображения")
        btn_load_images.clicked.connect(self.load_images)
        btn_load_images.setToolTip("Загрузка изображений для разметки")
        image_buttons_layout.addWidget(btn_load_images)
        btn_remove_image = QPushButton("Удалить изображение")
        btn_remove_image.clicked.connect(self.remove_image)
        btn_remove_image.setToolTip("Удаление выбранного изображения из проекта")
        image_buttons_layout.addWidget(btn_remove_image)
        right_layout.addLayout(image_buttons_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.thumbnails_container = QWidget()
        self.thumbnails_layout = QGridLayout(self.thumbnails_container)
        scroll_area.setWidget(self.thumbnails_container)
        right_layout.addWidget(scroll_area)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([700, 300])
        layout.addWidget(splitter)
        return tab
    def create_training_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        model_group = QGroupBox("Выбор модели")
        model_layout = QHBoxLayout(model_group)
        model_label = QLabel("Модель:")
        model_layout.addWidget(model_label)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLO", "MobileNet", "Пользовательская"])
        self.model_combo.setToolTip("Выберите тип модели для обучения")
        model_layout.addWidget(self.model_combo)
        btn_load_model = QPushButton("Загрузить модель")
        btn_load_model.clicked.connect(self.load_model)
        btn_load_model.setToolTip("Загрузка предобученной модели из файла или интернета")
        model_layout.addWidget(btn_load_model)
        top_layout.addWidget(model_group)
        training_group = QGroupBox("Настройки обучения")
        training_layout = QFormLayout(training_group)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        self.epochs_spin.setSuffix(" эпох")
        self.epochs_spin.setToolTip("Количество эпох обучения (проходов через весь датасет)")
        training_layout.addRow("Количество эпох:", self.epochs_spin)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(4)
        self.batch_size_spin.setToolTip("Количество изображений, обрабатываемых за одну итерацию")
        training_layout.addRow("Размер батча:", self.batch_size_spin)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.00001, 1.0)
        self.learning_rate_spin.setValue(0.001)
        self.learning_rate_spin.setDecimals(5)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.learning_rate_spin.setToolTip("Скорость обучения (learning rate) - влияет на скорость сходимости")
        training_layout.addRow("Скорость обучения:", self.learning_rate_spin)
        top_layout.addWidget(training_group)
        buttons_layout = QHBoxLayout()
        btn_create_dataset = QPushButton("Сформировать датасет")
        btn_create_dataset.clicked.connect(self.create_dataset)
        btn_create_dataset.setToolTip("Создание датасета из размеченных изображений")
        buttons_layout.addWidget(btn_create_dataset)
        self.btn_start_training = QPushButton("Запустить обучение")
        self.btn_start_training.clicked.connect(self.start_training)
        self.btn_start_training.setToolTip("Запуск процесса обучения модели на созданном датасете")
        buttons_layout.addWidget(self.btn_start_training)
        top_layout.addLayout(buttons_layout)
        layout.addWidget(top_panel)
        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)
        progress_label = QLabel("Прогресс обучения:")
        middle_layout.addWidget(progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setToolTip("Отображает прогресс обучения модели")
        middle_layout.addWidget(self.progress_bar)
        layout.addWidget(middle_panel)
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        log_label = QLabel("Лог обучения:")
        bottom_layout.addWidget(log_label)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setToolTip("Отображает информацию о процессе обучения")
        bottom_layout.addWidget(self.log_text)
        layout.addWidget(bottom_panel)
        return tab
    def create_testing_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        top_panel = QWidget()
        top_layout = QHBoxLayout(top_panel)
        btn_load_test_image = QPushButton("Загрузить тестовое изображение")
        btn_load_test_image.clicked.connect(self.load_test_image)
        btn_load_test_image.setToolTip("Загрузка изображения для тестирования модели")
        top_layout.addWidget(btn_load_test_image)
        self.btn_run_prediction = QPushButton("Запустить предсказание")
        self.btn_run_prediction.clicked.connect(self.run_prediction)
        self.btn_run_prediction.setEnabled(False)  
        self.btn_run_prediction.setToolTip("Запуск предсказания на тестовом изображении (требуется загруженная модель)")
        top_layout.addWidget(self.btn_run_prediction)
        layout.addWidget(top_panel)
        self.test_image_widget = ImageWidget()
        self.test_image_widget.setToolTip("Здесь отображается тестовое изображение с предсказаниями")
        layout.addWidget(self.test_image_widget)
        return tab
    def load_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Выберите изображения", "", "Изображения (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_paths:
            return
        self.clear_thumbnails()
        row, col = 0, 0
        for i, file_path in enumerate(file_paths):
            if file_path not in self.annotations:
                self.annotations[file_path] = ImageAnnotation(file_path)
            thumbnail_container = QWidget()
            thumbnail_layout = QVBoxLayout(thumbnail_container)
            thumbnail_layout.setContentsMargins(5, 5, 5, 5)
            thumbnail_layout.setSpacing(2)
            thumbnail = QLabel()
            pixmap = QPixmap(file_path).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
            thumbnail.setPixmap(pixmap)
            thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumbnail.setStyleSheet("border: 1px solid gray;")
            thumbnail_layout.addWidget(thumbnail)
            filename_label = QLabel(os.path.basename(file_path))
            filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            filename_label.setWordWrap(True)
            filename_label.setMaximumWidth(120)
            thumbnail_layout.addWidget(filename_label)
            thumbnail_container.mousePressEvent = lambda event, path=file_path: self.select_image(path)
            thumbnail_container.setToolTip(file_path)
            self.thumbnails_layout.addWidget(thumbnail_container, row, col)
            col += 1
            if col >= 3:  
                col = 0
                row += 1
        if file_paths:
            self.select_image(file_paths[0])
    def clear_thumbnails(self):
        for i in reversed(range(self.thumbnails_layout.count())):
            widget = self.thumbnails_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
    def remove_image(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите изображение для удаления")
            return
        reply = QMessageBox.question(
            self, 
            "Подтверждение", 
            f"Вы уверены, что хотите удалить изображение {os.path.basename(self.current_image_path)}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            if self.current_image_path in self.annotations:
                del self.annotations[self.current_image_path]
            next_image = None
            if self.annotations:
                next_image = next(iter(self.annotations.keys()))
            self.current_image_path = None
            self.update_thumbnails()
            if next_image:
                self.select_image(next_image)
            else:
                self.image_widget.setPixmap(QPixmap())
                self.image_widget.set_boxes([])
                self.selected_box_info.setText("Выберите объект для отображения информации")
    def select_image(self, image_path):
        self.current_image_path = image_path
        self.image_widget.set_image(image_path)
        if image_path in self.annotations:
            self.image_widget.set_boxes(self.annotations[image_path].boxes)
            if self.class_colors:
                self.image_widget.set_colors(self.class_colors)
    def on_box_added(self, x, y, width, height):
        if self.current_image_path and self.current_image_path in self.annotations:
            self.annotations[self.current_image_path].add_box(x, y, width, height, self.current_class_id)
            self.image_widget.set_boxes(self.annotations[self.current_image_path].boxes)
    def on_class_changed(self, index):
        self.current_class_id = index
    def add_class(self):
        class_names = [self.class_combo.itemText(i) for i in range(self.class_combo.count())]
        class_manager = ClassManager(class_names, self.class_colors, self)
        if class_manager.exec() == QDialog.DialogCode.Accepted:
            self.class_combo.clear()
            self.class_colors = []
            for i, class_name in enumerate(class_manager.class_names):
                self.class_combo.addItem(class_name)
                if i < len(class_manager.class_colors):
                    self.class_colors.append(class_manager.class_colors[i])
                else:
                    default_color_index = i % len(self.default_colors)
                    self.class_colors.append(self.default_colors[default_color_index])
            if self.class_combo.count() > 0:
                self.class_combo.setCurrentIndex(0)
            else:
                self.class_combo.addItem("Класс 0")
                self.class_colors.append(self.default_colors[0])
                self.class_combo.setCurrentIndex(0)
            self.image_widget.set_colors(self.class_colors)
    def on_box_selected(self, box_index):
        if self.current_image_path and self.current_image_path in self.annotations:
            annotation = self.annotations[self.current_image_path]
            if 0 <= box_index < len(annotation.boxes):
                box = annotation.boxes[box_index]
                x, y, w, h, class_id = box
                class_name = self.class_combo.itemText(class_id) if class_id < self.class_combo.count() else f"Класс {class_id}"
                self.selected_box_info.setText(
                    f"Выбран объект #{box_index}\n"
                    f"Класс: {class_name}\n"
                    f"Координаты: x={x}, y={y}, width={w}, height={h}"
                )
                if class_id < self.class_combo.count():
                    self.class_combo.setCurrentIndex(class_id)
    def on_box_edited(self, box_index, x, y, w, h):
        if self.current_image_path and self.current_image_path in self.annotations:
            annotation = self.annotations[self.current_image_path]
            if 0 <= box_index < len(annotation.boxes):
                class_id = annotation.boxes[box_index][4]
                annotation.boxes[box_index] = [x, y, w, h, class_id]
                self.image_widget.set_boxes(annotation.boxes)
                self.on_box_selected(box_index)
    def on_box_deleted(self, box_index):
        if self.current_image_path and self.current_image_path in self.annotations:
            annotation = self.annotations[self.current_image_path]
            if 0 <= box_index < len(annotation.boxes):
                annotation.remove_box(box_index)
                self.image_widget.set_boxes(annotation.boxes)
                self.selected_box_info.setText("Выберите объект для отображения информации")
    def remove_selected_box(self):
        if self.current_image_path and self.current_image_path in self.annotations:
            if hasattr(self.image_widget, 'selected_box_index') and self.image_widget.selected_box_index >= 0:
                self.on_box_deleted(self.image_widget.selected_box_index)
            elif self.annotations[self.current_image_path].boxes:
                last_index = len(self.annotations[self.current_image_path].boxes) - 1
                self.on_box_deleted(last_index)
    def create_new_project(self):
        directory = QFileDialog.getExistingDirectory(self, "Выберите директорию для нового проекта")
        if directory:
            self.annotations = {}
            self.current_image_path = None
            self.model_path = None
            self.project_dir = directory
            self.class_colors = []
            for i in range(self.class_combo.count()):
                self.class_colors.append(self.default_colors[i % len(self.default_colors)])
            os.makedirs(os.path.join(directory, "images"), exist_ok=True)
            os.makedirs(os.path.join(directory, "annotations"), exist_ok=True)
            os.makedirs(os.path.join(directory, "models"), exist_ok=True)
            os.makedirs(os.path.join(directory, "datasets"), exist_ok=True)
            self.setWindowTitle(f"NeuroMark - {os.path.basename(directory)}")
            self.clear_thumbnails()
            self.image_widget.setPixmap(QPixmap())
            self.image_widget.set_boxes([])
            self.log_text.clear()
            self.progress_bar.setValue(0)
    def save_project(self):
        if not self.project_dir:
            QMessageBox.warning(self, "Ошибка", "Сначала создайте или загрузите проект.")
            return
        try:
            annotations_data = {}
            for image_path, annotation in self.annotations.items():
                if not image_path.startswith(self.project_dir):
                    target_path = os.path.join(self.project_dir, "images", os.path.basename(image_path))
                    if not os.path.exists(target_path):
                        import shutil
                        shutil.copy2(image_path, target_path)
                    annotation.image_path = target_path
                    annotations_data[target_path] = annotation.to_json()
                else:
                    annotations_data[image_path] = annotation.to_json()
            annotations_file = os.path.join(self.project_dir, "annotations", "annotations.json")
            with open(annotations_file, "w") as f:
                json.dump(annotations_data, f, indent=4)
            class_colors_data = []
            for color in self.class_colors:
                class_colors_data.append({
                    "r": color.red(),
                    "g": color.green(),
                    "b": color.blue(),
                    "a": color.alpha()
                })
            project_info = {
                "name": os.path.basename(self.project_dir),
                "classes": [self.class_combo.itemText(i) for i in range(self.class_combo.count())],
                "class_colors": class_colors_data,
                "model_path": self.model_path
            }
            project_file = os.path.join(self.project_dir, "project.json")
            with open(project_file, "w") as f:
                json.dump(project_info, f, indent=4)
            QMessageBox.information(self, "Успех", "Проект успешно сохранен.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить проект: {str(e)}")
    def load_project(self):
        directory = QFileDialog.getExistingDirectory(self, "Выберите директорию проекта")
        if not directory:
            return
        project_file = os.path.join(directory, "project.json")
        if not os.path.exists(project_file):
            QMessageBox.warning(self, "Ошибка", "Выбранная директория не содержит файла проекта.")
            return
        try:
            with open(project_file, "r") as f:
                project_info = json.load(f)
            self.project_dir = directory
            self.setWindowTitle(f"NeuroMark - {project_info['name']}")
            self.class_combo.clear()
            for class_name in project_info["classes"]:
                self.class_combo.addItem(class_name)
            self.model_path = project_info.get("model_path")
            annotations_file = os.path.join(directory, "annotations", "annotations.json")
            if os.path.exists(annotations_file):
                with open(annotations_file, "r") as f:
                    annotations_data = json.load(f)
                self.annotations = {}
                for image_path, annotation_data in annotations_data.items():
                    if os.path.exists(image_path):
                        self.annotations[image_path] = ImageAnnotation.from_json(annotation_data)
            self.update_thumbnails()
            QMessageBox.information(self, "Успех", "Проект успешно загружен.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить проект: {str(e)}")
    def update_thumbnails(self):
        self.clear_thumbnails()
        row, col = 0, 0
        for image_path in self.annotations.keys():
            thumbnail_container = QWidget()
            thumbnail_layout = QVBoxLayout(thumbnail_container)
            thumbnail_layout.setContentsMargins(5, 5, 5, 5)
            thumbnail_layout.setSpacing(2)
            thumbnail = QLabel()
            pixmap = QPixmap(image_path).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
            thumbnail.setPixmap(pixmap)
            thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumbnail.setStyleSheet("border: 1px solid gray;")
            thumbnail_layout.addWidget(thumbnail)
            filename_label = QLabel(os.path.basename(image_path))
            filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            filename_label.setWordWrap(True)
            filename_label.setMaximumWidth(120)
            thumbnail_layout.addWidget(filename_label)
            thumbnail_container.mousePressEvent = lambda event, path=image_path: self.select_image(path)
            thumbnail_container.setToolTip(image_path)
            self.thumbnails_layout.addWidget(thumbnail_container, row, col)
            col += 1
            if col >= 3:  
                col = 0
                row += 1
        if self.annotations:
            self.select_image(next(iter(self.annotations.keys())))
    def load_model(self):
        source_dialog = QDialog(self)
        source_dialog.setWindowTitle("Загрузка модели")
        source_dialog.setMinimumWidth(400)
        layout = QVBoxLayout(source_dialog)
        layout.addWidget(QLabel("Выберите источник модели:"))
        btn_local = QPushButton("Загрузить локальную модель")
        btn_local.clicked.connect(lambda: self.load_local_model(source_dialog))
        layout.addWidget(btn_local)
        btn_internet = QPushButton("Загрузить модель из интернета")
        btn_internet.clicked.connect(lambda: self.load_internet_model(source_dialog))
        layout.addWidget(btn_internet)
        btn_cancel = QPushButton("Отмена")
        btn_cancel.clicked.connect(source_dialog.reject)
        layout.addWidget(btn_cancel)
        source_dialog.exec()
    def load_local_model(self, parent_dialog=None):
        if parent_dialog:
            parent_dialog.accept()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл модели", "", "Модели PyTorch (*.pt *.pth)"
        )
        if not file_path:
            return
        try:
            model_data = torch.load(file_path, map_location=torch.device('cpu'))
            if not self.validate_model(model_data):
                reply = QMessageBox.question(
                    self, 
                    "Предупреждение", 
                    "Модель может быть несовместима с текущим датасетом. Продолжить загрузку?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
            self.model_path = file_path
            model_info = self.get_model_info(model_data)
            QMessageBox.information(
                self, 
                "Успех", 
                f"Модель {os.path.basename(file_path)} успешно загружена.\n\n{model_info}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель: {str(e)}")
    def load_internet_model(self, parent_dialog=None):
        if parent_dialog:
            parent_dialog.accept()
        model_dialog = QDialog(self)
        model_dialog.setWindowTitle("Загрузка модели из интернета")
        model_dialog.setMinimumSize(500, 400)
        layout = QVBoxLayout(model_dialog)
        layout.addWidget(QLabel("Доступные модели:"))
        model_list = QListWidget()
        layout.addWidget(model_list)
        available_models = ModelDownloader.get_available_models()
        for model_name, model_info in available_models.items():
            item = QListWidgetItem(f"{model_name} - {model_info['description']}")
            item.setData(Qt.ItemDataRole.UserRole, model_name)
            model_list.addItem(item)
        info_label = QLabel("Выберите модель из списка")
        layout.addWidget(info_label)
        progress_bar = QProgressBar()
        progress_bar.setVisible(False)
        layout.addWidget(progress_bar)
        buttons_layout = QHBoxLayout()
        btn_download = QPushButton("Загрузить")
        btn_download.setEnabled(False)
        buttons_layout.addWidget(btn_download)
        btn_cancel = QPushButton("Отмена")
        btn_cancel.clicked.connect(model_dialog.reject)
        buttons_layout.addWidget(btn_cancel)
        layout.addLayout(buttons_layout)
        def on_model_selected():
            selected_items = model_list.selectedItems()
            if selected_items:
                model_name = selected_items[0].data(Qt.ItemDataRole.UserRole)
                model_info = available_models[model_name]
                info_label.setText(
                    f"Модель: {model_name}\n"
                    f"Тип: {model_info['type']}\n"
                    f"Описание: {model_info['description']}"
                )
                btn_download.setEnabled(True)
            else:
                info_label.setText("Выберите модель из списка")
                btn_download.setEnabled(False)
        model_list.itemSelectionChanged.connect(on_model_selected)
        def on_download_clicked():
            selected_items = model_list.selectedItems()
            if not selected_items:
                return
            model_name = selected_items[0].data(Qt.ItemDataRole.UserRole)
            if not self.project_dir:
                QMessageBox.warning(self, "Ошибка", "Сначала создайте или загрузите проект.")
                return
            models_dir = os.path.join(self.project_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            btn_download.setEnabled(False)
            btn_cancel.setEnabled(False)
            model_list.setEnabled(False)
            progress_bar.setVisible(True)
            progress_bar.setValue(0)
            def update_progress(value):
                progress_bar.setValue(value)
                QApplication.processEvents()  
            try:
                model_path = ModelDownloader.download_model(model_name, models_dir, update_progress)
                self.model_path = model_path
                model_dialog.accept()
                QMessageBox.information(
                    self, 
                    "Успех", 
                    f"Модель {model_name} успешно загружена в {model_path}"
                )
            except Exception as e:
                btn_download.setEnabled(True)
                btn_cancel.setEnabled(True)
                model_list.setEnabled(True)
                progress_bar.setVisible(False)
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель: {str(e)}")
        btn_download.clicked.connect(on_download_clicked)
        model_dialog.exec()
    def validate_model(self, model_data):
        if isinstance(model_data, dict):
            if "model_type" in model_data:
                num_classes = model_data.get("num_classes", 0)
                if num_classes > 0 and num_classes != self.class_combo.count():
                    return False
                return True
            if "state_dict" in model_data or any(key.endswith(".weight") for key in model_data.keys()):
                return True
        if hasattr(model_data, "state_dict"):
            return True
        return False
    def get_model_info(self, model_data):
        info = []
        if isinstance(model_data, dict):
            if "model_type" in model_data:
                info.append(f"Тип модели: {model_data.get('model_type', 'Неизвестно')}")
                info.append(f"Количество классов: {model_data.get('num_classes', 'Неизвестно')}")
                class_names = model_data.get("class_names", [])
                if class_names:
                    info.append("Классы:")
                    for i, name in enumerate(class_names):
                        info.append(f"  {i}: {name}")
                if "source" in model_data:
                    info.append(f"Источник: {model_data['source']}")
                    if "url" in model_data:
                        info.append(f"URL: {model_data['url']}")
            else:
                info.append("Тип модели: PyTorch state_dict")
                info.append(f"Количество параметров: {len(model_data)}")
        else:
            info.append("Тип модели: PyTorch модель")
        return "\n".join(info)
    def create_dataset(self, annotation_format="YOLO"):
        if not self.project_dir:
            QMessageBox.warning(self, "Ошибка", "Сначала создайте или загрузите проект.")
            return
        if not self.annotations:
            QMessageBox.warning(self, "Ошибка", "Нет размеченных изображений для создания датасета.")
            return
        formats = ["YOLO", "COCO", "JSON"]
        selected_format, ok = QInputDialog.getItem(
            self, "Выбор формата аннотаций", 
            "Выберите формат аннотаций:", 
            formats, 0, False
        )
        if not ok:
            return
        try:
            dataset_dir = os.path.join(self.project_dir, "datasets", 
                                      f"dataset_{selected_format.lower()}_{time.strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(dataset_dir, exist_ok=True)
            train_dir = os.path.join(dataset_dir, "train")
            val_dir = os.path.join(dataset_dir, "val")
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            image_paths = list(self.annotations.keys())
            random.shuffle(image_paths)
            split_idx = int(len(image_paths) * 0.8)  
            train_images = image_paths[:split_idx]
            val_images = image_paths[split_idx:]
            self.copy_images_and_annotations(train_images, train_dir, selected_format)
            self.copy_images_and_annotations(val_images, val_dir, selected_format)
            classes_file = os.path.join(dataset_dir, "classes.txt")
            with open(classes_file, "w") as f:
                for i in range(self.class_combo.count()):
                    f.write(f"{i}: {self.class_combo.itemText(i)}\n")
            readme_file = os.path.join(dataset_dir, "README.md")
            with open(readme_file, "w") as f:
                f.write(f"# Датасет для обучения модели обнаружения объектов\n\n")
                f.write(f"Создан: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Формат аннотаций: {selected_format}\n\n")
                f.write(f"## Классы объектов\n\n")
                for i in range(self.class_combo.count()):
                    f.write(f"- {i}: {self.class_combo.itemText(i)}\n")
                f.write(f"\n## Статистика\n\n")
                f.write(f"- Всего изображений: {len(image_paths)}\n")
                f.write(f"- Обучающая выборка: {len(train_images)}\n")
                f.write(f"- Валидационная выборка: {len(val_images)}\n")
            QMessageBox.information(
                self, 
                "Успех", 
                f"Датасет успешно создан в {dataset_dir}\nФормат аннотаций: {selected_format}"
            )
            return dataset_dir
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось создать датасет: {str(e)}")
            return None
    def copy_images_and_annotations(self, image_paths, target_dir, format="YOLO"):
        images_dir = os.path.join(target_dir, "images")
        labels_dir = os.path.join(target_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        if format.upper() == "COCO":
            coco_annotations = {
                "images": [],
                "annotations": [],
                "categories": []
            }
            for i in range(self.class_combo.count()):
                category = {
                    "id": i,
                    "name": self.class_combo.itemText(i),
                    "supercategory": "object"
                }
                coco_annotations["categories"].append(category)
        image_id = 0
        annotation_id = 0
        for image_path in image_paths:
            image_filename = os.path.basename(image_path)
            target_image_path = os.path.join(images_dir, image_filename)
            shutil.copy2(image_path, target_image_path)
            annotation = self.annotations[image_path]
            if format.upper() == "YOLO":
                label_filename = os.path.splitext(image_filename)[0] + ".txt"
                label_path = os.path.join(labels_dir, label_filename)
                yolo_annotations = annotation.to_yolo_format()
                with open(label_path, "w") as f:
                    for line in yolo_annotations:
                        f.write(line + "\n")
            elif format.upper() == "COCO":
                image_info, annotations_list = annotation.to_coco_format(image_id)
                image_info["file_name"] = image_filename  
                coco_annotations["images"].append(image_info)
                for ann in annotations_list:
                    ann["id"] = annotation_id
                    coco_annotations["annotations"].append(ann)
                    annotation_id += 1
                image_id += 1
            else:
                label_filename = os.path.splitext(image_filename)[0] + ".json"
                label_path = os.path.join(labels_dir, label_filename)
                with open(label_path, "w") as f:
                    json.dump(annotation.to_json(), f, indent=4)
        if format.upper() == "COCO":
            coco_file = os.path.join(target_dir, "annotations.json")
            with open(coco_file, "w") as f:
                json.dump(coco_annotations, f, indent=4)
    def start_training(self):
        if not self.project_dir:
            QMessageBox.warning(self, "Ошибка", "Сначала создайте или загрузите проект.")
            return
        dataset_dir = self.create_dataset()
        if not dataset_dir:
            return
        model_type = self.model_combo.currentText()
        self.training_thread = TrainingThread(model_type, dataset_dir)
        self.training_thread.progress_updated.connect(self.update_progress)
        self.training_thread.log_updated.connect(self.update_log)
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.btn_start_training.setEnabled(False)
        self.btn_start_training.setText("Обучение...")
        self.training_thread.start()
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    def update_log(self, message):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    def on_training_finished(self, model_path):
        self.model_path = model_path
        self.btn_start_training.setEnabled(True)
        self.btn_start_training.setText("Запустить обучение")
        self.update_log(f"Обучение завершено. Модель сохранена в {model_path}")
        QMessageBox.information(self, "Успех", "Обучение модели успешно завершено!")
    def load_test_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите тестовое изображение", "", "Изображения (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return
        self.test_image_widget.set_image(file_path)
        self.test_image_path = file_path
        if self.class_colors:
            self.test_image_widget.set_colors(self.class_colors)
        self.btn_run_prediction.setEnabled(self.model_path is not None)
    def run_prediction(self):
        if not self.model_path or not hasattr(self, 'test_image_path'):
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите модель и тестовое изображение.")
            return
        try:
            self.log_text.append(f"Загрузка модели из {self.model_path}...")
            model_info = torch.load(self.model_path, map_location=torch.device('cpu'))
            model_type = model_info.get("model_type", "Неизвестно")
            num_classes = model_info.get("num_classes", 1)
            class_names = model_info.get("class_names", [])
            self.log_text.append(f"Тип модели: {model_type}")
            self.log_text.append(f"Количество классов: {num_classes}")
            if model_type == "MobileNet" or model_type == "YOLO" or model_type == "Пользовательская":
                model = SimpleObjectDetector(num_classes)
            else:
                raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
            model.load_state_dict(model_info["model_state_dict"])
            model.eval()  
            transform_params = model_info.get("transform_params", {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            })
            transform = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=transform_params.get("mean", [0.485, 0.456, 0.406]),
                    std=transform_params.get("std", [0.229, 0.224, 0.225])
                )
            ])
            image = Image.open(self.test_image_path).convert("RGB")
            img_width, img_height = image.size
            input_tensor = transform(image).unsqueeze(0)  
            with torch.no_grad():
                outputs = model(input_tensor)
            pred_boxes = outputs['boxes'][0].cpu().numpy()
            pred_scores = outputs['scores'][0].cpu().numpy()
            confidence_threshold = 0.3
            boxes = []
            img_cv = cv2.imread(self.test_image_path)
            height, width = img_cv.shape[:2]
            for i in range(min(20, len(pred_boxes))):  
                max_score = np.max(pred_scores[i])
                if max_score > confidence_threshold:
                    class_id = np.argmax(pred_scores[i])
                    box = pred_boxes[i]
                    x1, y1, x2, y2 = box
                    x = int(x1 * width / 300)
                    y = int(y1 * height / 300)
                    w = int((x2 - x1) * width / 300)
                    h = int((y2 - y1) * height / 300)
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    w = max(1, min(w, width - x))
                    h = max(1, min(h, height - y))
                    boxes.append([x, y, w, h, class_id])
            self.test_image_widget.set_boxes(boxes)
            if boxes:
                self.log_text.append(f"Обнаружено объектов: {len(boxes)}")
                for i, box in enumerate(boxes):
                    x, y, w, h, class_id = box
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Класс {class_id}"
                    self.log_text.append(f"Объект {i+1}: {class_name}, координаты: ({x}, {y}, {w}, {h})")
                QMessageBox.information(self, "Успех", f"Обнаружено {len(boxes)} объектов.")
            else:
                self.log_text.append("Объекты не обнаружены.")
                QMessageBox.information(self, "Результат", "Объекты не обнаружены.")
        except Exception as e:
            import traceback
            self.log_text.append(f"Ошибка при выполнении предсказания: {str(e)}")
            self.log_text.append(traceback.format_exc())
            QMessageBox.critical(self, "Ошибка", f"Ошибка при выполнении предсказания: {str(e)}")
class ImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Импорт разметок")
        self.resize(500, 400)
        self.parent = parent
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout(self)
        tab_widget = QTabWidget()
        yolo_tab = QWidget()
        yolo_layout = QVBoxLayout(yolo_tab)
        yolo_description = QLabel("Импорт разметок в формате YOLO.")
        yolo_description.setWordWrap(True)
        yolo_layout.addWidget(yolo_description)
        yolo_form = QFormLayout()
        self.yolo_images_dir = QLineEdit()
        self.yolo_images_dir.setReadOnly(True)
        yolo_images_btn = QPushButton("Обзор...")
        yolo_images_btn.clicked.connect(self.browse_yolo_images)
        yolo_images_layout = QHBoxLayout()
        yolo_images_layout.addWidget(self.yolo_images_dir)
        yolo_images_layout.addWidget(yolo_images_btn)
        yolo_form.addRow("Директория с изображениями:", yolo_images_layout)
        self.yolo_labels_dir = QLineEdit()
        self.yolo_labels_dir.setReadOnly(True)
        yolo_labels_btn = QPushButton("Обзор...")
        yolo_labels_btn.clicked.connect(self.browse_yolo_labels)
        yolo_labels_layout = QHBoxLayout()
        yolo_labels_layout.addWidget(self.yolo_labels_dir)
        yolo_labels_layout.addWidget(yolo_labels_btn)
        yolo_form.addRow("Директория с разметками YOLO:", yolo_labels_layout)
        yolo_layout.addLayout(yolo_form)
        yolo_import_btn = QPushButton("Импортировать YOLO разметки")
        yolo_import_btn.clicked.connect(self.import_yolo)
        yolo_layout.addWidget(yolo_import_btn)
        tab_widget.addTab(yolo_tab, "YOLO")
        coco_tab = QWidget()
        coco_layout = QVBoxLayout(coco_tab)
        coco_description = QLabel("Импорт разметок в формате COCO.")
        coco_description.setWordWrap(True)
        coco_layout.addWidget(coco_description)
        coco_form = QFormLayout()
        self.coco_images_dir = QLineEdit()
        self.coco_images_dir.setReadOnly(True)
        coco_images_btn = QPushButton("Обзор...")
        coco_images_btn.clicked.connect(self.browse_coco_images)
        coco_images_layout = QHBoxLayout()
        coco_images_layout.addWidget(self.coco_images_dir)
        coco_images_layout.addWidget(coco_images_btn)
        coco_form.addRow("Директория с изображениями:", coco_images_layout)
        self.coco_annotation_file = QLineEdit()
        self.coco_annotation_file.setReadOnly(True)
        coco_annotation_btn = QPushButton("Обзор...")
        coco_annotation_btn.clicked.connect(self.browse_coco_annotation)
        coco_annotation_layout = QHBoxLayout()
        coco_annotation_layout.addWidget(self.coco_annotation_file)
        coco_annotation_layout.addWidget(coco_annotation_btn)
        coco_form.addRow("Файл аннотаций COCO:", coco_annotation_layout)
        coco_layout.addLayout(coco_form)
        coco_import_btn = QPushButton("Импортировать COCO разметки")
        coco_import_btn.clicked.connect(self.import_coco)
        coco_layout.addWidget(coco_import_btn)
        tab_widget.addTab(coco_tab, "COCO")
        json_tab = QWidget()
        json_layout = QVBoxLayout(json_tab)
        json_description = QLabel("Импорт разметок в собственном JSON формате приложения.")
        json_description.setWordWrap(True)
        json_layout.addWidget(json_description)
        json_form = QFormLayout()
        self.json_annotation_file = QLineEdit()
        self.json_annotation_file.setReadOnly(True)
        json_annotation_btn = QPushButton("Обзор...")
        json_annotation_btn.clicked.connect(self.browse_json_annotation)
        json_annotation_layout = QHBoxLayout()
        json_annotation_layout.addWidget(self.json_annotation_file)
        json_annotation_layout.addWidget(json_annotation_btn)
        json_form.addRow("Файл аннотаций JSON:", json_annotation_layout)
        json_layout.addLayout(json_form)
        json_import_btn = QPushButton("Импортировать JSON разметки")
        json_import_btn.clicked.connect(self.import_json)
        json_layout.addWidget(json_import_btn)
        tab_widget.addTab(json_tab, "JSON")
        layout.addWidget(tab_widget)
        self.status_label = QLabel("Готов к импорту")
        layout.addWidget(self.status_label)
        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
    def browse_yolo_images(self):
        directory = QFileDialog.getExistingDirectory(self, "Выберите директорию с изображениями")
        if directory:
            self.yolo_images_dir.setText(directory)
    def browse_yolo_labels(self):
        directory = QFileDialog.getExistingDirectory(self, "Выберите директорию с разметками YOLO")
        if directory:
            self.yolo_labels_dir.setText(directory)
    def browse_coco_images(self):
        directory = QFileDialog.getExistingDirectory(self, "Выберите директорию с изображениями")
        if directory:
            self.coco_images_dir.setText(directory)
    def browse_coco_annotation(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл аннотаций COCO", "", "JSON файлы (*.json)")
        if file_path:
            self.coco_annotation_file.setText(file_path)
    def browse_json_annotation(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл аннотаций JSON", "", "JSON файлы (*.json)")
        if file_path:
            self.json_annotation_file.setText(file_path)
    def import_yolo(self):
        images_dir = self.yolo_images_dir.text()
        labels_dir = self.yolo_labels_dir.text()
        if not images_dir or not labels_dir:
            QMessageBox.warning(self, "Ошибка", "Укажите директории с изображениями и разметками")
            return
        try:
            image_files = [f for f in os.listdir(images_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if not image_files:
                QMessageBox.warning(self, "Ошибка", "В указанной директории не найдены изображения")
                return
            imported_count = 0
            for image_file in image_files:
                image_path = os.path.join(images_dir, image_file)
                label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + ".txt")
                if os.path.exists(label_path):
                    annotation = ImageAnnotation.from_yolo_file(image_path, label_path)
                    if hasattr(self.parent, 'annotations'):
                        self.parent.annotations[image_path] = annotation
                        imported_count += 1
            if hasattr(self.parent, 'update_thumbnails'):
                self.parent.update_thumbnails()
            self.status_label.setText(f"Импортировано {imported_count} изображений с разметками YOLO")
            if imported_count > 0:
                QMessageBox.information(self, "Успех", f"Импортировано {imported_count} изображений с разметками YOLO")
            else:
                QMessageBox.warning(self, "Предупреждение", "Не найдены соответствующие разметки для изображений")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при импорте YOLO разметок: {str(e)}")
            self.status_label.setText(f"Ошибка: {str(e)}")
    def import_coco(self):
        images_dir = self.coco_images_dir.text()
        annotation_file = self.coco_annotation_file.text()
        if not images_dir or not annotation_file:
            QMessageBox.warning(self, "Ошибка", "Укажите директорию с изображениями и файл аннотаций")
            return
        try:
            with open(annotation_file, 'r') as f:
                coco_data = json.load(f)
            if not coco_data:
                QMessageBox.warning(self, "Ошибка", "Файл аннотаций COCO пуст или имеет неверный формат")
                return
            if 'images' not in coco_data or 'annotations' not in coco_data:
                QMessageBox.warning(self, "Ошибка", "Файл аннотаций COCO имеет неверный формат")
                return
            image_id_map = {}
            for img in coco_data['images']:
                if 'id' in img and 'file_name' in img:
                    image_id_map[img['file_name']] = img['id']
            imported_count = 0
            for image_file, image_id in image_id_map.items():
                image_path = os.path.join(images_dir, image_file)
                if os.path.exists(image_path):
                    annotation = ImageAnnotation.from_coco_annotation(image_path, coco_data, image_id)
                    if annotation.boxes:
                        if hasattr(self.parent, 'annotations'):
                            self.parent.annotations[image_path] = annotation
                            imported_count += 1
            if hasattr(self.parent, 'update_thumbnails'):
                self.parent.update_thumbnails()
            self.status_label.setText(f"Импортировано {imported_count} изображений с разметками COCO")
            if imported_count > 0:
                QMessageBox.information(self, "Успех", f"Импортировано {imported_count} изображений с разметками COCO")
            else:
                QMessageBox.warning(self, "Предупреждение", "Не найдены соответствующие изображения для разметок COCO")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при импорте COCO разметок: {str(e)}")
            self.status_label.setText(f"Ошибка: {str(e)}")
    def import_json(self):
        annotation_file = self.json_annotation_file.text()
        if not annotation_file:
            QMessageBox.warning(self, "Ошибка", "Укажите файл аннотаций JSON")
            return
        try:
            with open(annotation_file, 'r') as f:
                annotations_data = json.load(f)
            if not annotations_data:
                QMessageBox.warning(self, "Ошибка", "Файл аннотаций JSON пуст или имеет неверный формат")
                return
            imported_count = 0
            for image_path, annotation_data in annotations_data.items():
                if os.path.exists(image_path):
                    annotation = ImageAnnotation.from_json(annotation_data)
                    if hasattr(self.parent, 'annotations'):
                        self.parent.annotations[image_path] = annotation
                        imported_count += 1
            if hasattr(self.parent, 'update_thumbnails'):
                self.parent.update_thumbnails()
            self.status_label.setText(f"Импортировано {imported_count} изображений с разметками JSON")
            if imported_count > 0:
                QMessageBox.information(self, "Успех", f"Импортировано {imported_count} изображений с разметками JSON")
            else:
                QMessageBox.warning(self, "Предупреждение", "Не найдены соответствующие изображения для разметок JSON")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при импорте JSON разметок: {str(e)}")
            self.status_label.setText(f"Ошибка: {str(e)}")
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 
