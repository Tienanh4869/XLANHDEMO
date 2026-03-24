import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic
from PyQt5.QtCore import Qt
from giaodien import Ui_MainWindow  # Đảm bảo file giaodien.py tồn tại


class ManHinhChao(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.BatDau)

    def BatDau(self):
        self.main_app = ImageProcessorApp()
        self.main_app.show()
        self.close()


class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # 1. Load file UI
        uic.loadUi('UI_FixIMG.ui', self)

        # 2. Khởi tạo biến
        self.original_img = None  # Ảnh gốc không đổi để làm mốc
        self.current_img = None  # Ảnh sau khi xử lý slider
        self.history = []  # Lịch sử cho các thao tác không đảo ngược (Rotate, Crop, Flip)

        # 3. Kết nối Menu Actions
        self.actionOpen.triggered.connect(self.open_image)
        self.actionSave.triggered.connect(self.save_image)

        # 4. Kết nối Sliders (Xử lý cộng dồn)
        self.sliderGamma.valueChanged.connect(self.apply_all_effects)
        self.SliderDenoise.valueChanged.connect(self.apply_all_effects)
        self.SliderLammin.valueChanged.connect(self.apply_all_effects)

        # 5. Kết nối Buttons
        self.btnHistEq.clicked.connect(self.apply_hist_eq)
        self.btnRotate.clicked.connect(self.rotate_image)
        self.btnFlip.clicked.connect(self.flip_image)
        self.btnCrop.clicked.connect(self.crop_image)
        self.btnUndo.clicked.connect(self.undo_action)
        self.btnReset.clicked.connect(self.reset_image)

    # --- HIỂN THỊ ---
    def display_image(self, img, label):
        if img is None: return
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qt_img)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))

    # --- FILE OPS ---
    def open_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg *.bmp)",
                                                   options=options)
        if file_name:
            self.original_img = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_COLOR)
            if self.original_img is None:
                QMessageBox.warning(self, "Lỗi", "Không thể đọc được file ảnh!")
                return

            self.reset_ui_values()
            self.current_img = self.original_img.copy()
            self.history.clear()
            self.display_image(self.original_img, self.lblOriginal)
            self.display_image(self.current_img, self.lblProcessed)

    def save_image(self):
        if self.current_img is None: return
        file_name, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh", "", "JPEG (*.jpg);;PNG (*.png)")
        if file_name:
            cv2.imwrite(file_name, self.current_img)
            QMessageBox.information(self, "Thành công", "Đã lưu ảnh!")

    # --- LOGIC CỐT LÕI: CỘNG DỒN HIỆU ỨNG ---
    def apply_all_effects(self):
        """Mỗi khi kéo slider, hàm này chạy lại từ đầu dựa trên original_img"""
        if self.original_img is None: return

        # Bắt đầu từ bản sao của ảnh gốc hiện tại
        temp_img = self.original_img.copy()

        # 1. Xử lý Gamma
        gamma_val = self.sliderGamma.value()
        gamma = gamma_val / 50.0
        if gamma == 0: gamma = 0.01
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        temp_img = cv2.LUT(temp_img, table)

        # 2. Xử lý Khử nhiễu & Hiển thị %
        denoise_val = self.SliderDenoise.value()
        if hasattr(self, 'lblDenoiseValue'):
            self.lblDenoiseValue.setText(f"{denoise_val}%")

        if denoise_val > 0:
            k_size = (denoise_val // 10) * 2 + 1  # Tạo số lẻ 1, 3, 5...
            temp_img = cv2.GaussianBlur(temp_img, (k_size, k_size), 0)

        # 3. Xử lý Làm mịn & Hiển thị %
        smooth_val = self.SliderLammin.value()
        if hasattr(self, 'lblSmoothValue'):
            self.lblSmoothValue.setText(f"{smooth_val}%")

        if smooth_val > 0:
            d = max(1, smooth_val // 5)
            sigma = smooth_val * 2
            temp_img = cv2.bilateralFilter(temp_img, d, sigma, sigma)

        self.current_img = temp_img
        self.display_image(self.current_img, self.lblProcessed)

    # --- THAO TÁC BIẾN ĐỔI HÌNH HỌC (Có lưu History) ---
    def rotate_image(self):
        if self.original_img is None: return
        self.save_state()
        self.original_img = cv2.rotate(self.original_img, cv2.ROTATE_90_CLOCKWISE)
        self.apply_all_effects()

    def flip_image(self):
        if self.original_img is None: return
        self.save_state()
        self.original_img = cv2.flip(self.original_img, 1)
        self.apply_all_effects()

    def crop_image(self):
        if self.original_img is None: return
        self.save_state()
        h, w = self.original_img.shape[:2]
        self.original_img = self.original_img[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
        self.apply_all_effects()

    def apply_hist_eq(self):
        if self.original_img is None: return
        self.save_state()
        img_yuv = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        self.original_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        self.apply_all_effects()

    # --- HỆ THỐNG BỔ TRỢ ---
    def save_state(self):
        if self.original_img is not None:
            if len(self.history) >= 5: self.history.pop(0)
            self.history.append(self.original_img.copy())

    def undo_action(self):
        if self.history:
            self.original_img = self.history.pop()
            self.apply_all_effects()

    def reset_ui_values(self):
        """Đưa các thanh trượt về vị trí mặc định mà không kích hoạt xử lý thừa"""
        self.sliderGamma.blockSignals(True)
        self.SliderDenoise.blockSignals(True)
        self.SliderLammin.blockSignals(True)

        self.sliderGamma.setValue(50)
        self.SliderDenoise.setValue(0)
        self.SliderLammin.setValue(0)

        if hasattr(self, 'lblDenoiseValue'): self.lblDenoiseValue.setText("0%")
        if hasattr(self, 'lblSmoothValue'): self.lblSmoothValue.setText("0%")

        self.sliderGamma.blockSignals(False)
        self.SliderDenoise.blockSignals(False)
        self.SliderLammin.blockSignals(False)

    def reset_image(self):
        if self.original_img is not None:
            self.reset_ui_values()
            # Ở đây bạn có thể chọn reset về ảnh lúc mới Open hoặc giữ nguyên ảnh đã Rotate/Crop
            # Thông thường Reset là xóa hết filter (Gamma/Denoise/Smooth)
            self.apply_all_effects()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ManHinhChao()
    window.show()
    sys.exit(app.exec_())