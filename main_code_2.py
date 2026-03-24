import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QDialog, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic
from PyQt5.QtCore import Qt
from hihi import Ui_Form

# --- THƯ VIỆN VẼ BIỂU ĐỒ MATPLOTLIB ---
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ManHinhChao(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.BatDau)

    def BatDau(self):
        self.main_app = ImageProcessorApp()
        self.main_app.show()
        self.close()


# --- CLASS CỬA SỔ HIỂN THỊ HISTOGRAM CHUYÊN SÂU (UPDATE) ---
class HistogramDialog(QDialog):
    def __init__(self, orig_img, curr_img, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Phân tích và So sánh Histogram")
        self.resize(1100, 700)  # Mở rộng cửa sổ để chứa nhiều thông tin hơn

        # Chuyển đổi sang ảnh xám (Grayscale) để tính toán giống mẫu
        self.orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        self.proc_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

        # Tạo Layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Tạo Figure của Matplotlib
        fig = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)

        # Sử dụng GridSpec để chia bố cục 2 hàng x 3 cột
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])

        # Khởi tạo các vùng hiển thị (Subplots)
        ax_img_orig = fig.add_subplot(gs[0, 0])  # Hàng trên, Cột 1: Ảnh gốc
        ax_hist_orig = fig.add_subplot(gs[0, 1])  # Hàng trên, Cột 2: Hist gốc
        ax_img_proc = fig.add_subplot(gs[0, 2])  # Hàng trên, Cột 3: Ảnh xử lý
        ax_hist_comp = fig.add_subplot(gs[1, 0:2])  # Hàng dưới, Cột 1+2: So sánh Hist
        ax_stats = fig.add_subplot(gs[1, 2])  # Hàng dưới, Cột 3: Thống kê chữ

        # Tính toán dữ liệu Histogram
        hist_orig = cv2.calcHist([self.orig_gray], [0], None, [256], [0, 256]).flatten()
        hist_proc = cv2.calcHist([self.proc_gray], [0], None, [256], [0, 256]).flatten()

        # 1. Hiển thị Ảnh gốc
        ax_img_orig.imshow(self.orig_gray, cmap='gray')
        ax_img_orig.set_title('Ảnh gốc', fontweight='bold')
        ax_img_orig.axis('off')  # Tắt trục tọa độ cho ảnh

        # 2. Hiển thị Histogram gốc (Tô đen toàn bộ giống mẫu)
        ax_hist_orig.fill_between(range(256), hist_orig, color='black')
        ax_hist_orig.set_title('Histogram gốc', fontweight='bold')
        ax_hist_orig.set_ylabel('Số pixel')
        ax_hist_orig.set_xlabel('Mức xám')
        ax_hist_orig.set_xlim([0, 256])
        ax_hist_orig.grid(True, linestyle='-', alpha=0.3)

        # 3. Hiển thị Ảnh đã xử lý
        ax_img_proc.imshow(self.proc_gray, cmap='gray')
        ax_img_proc.set_title('Ảnh sau khi xử lý', fontweight='bold')
        ax_img_proc.axis('off')

        # 4. Hiển thị So sánh Histogram (Đường đỏ & Cột xanh giống mẫu)
        ax_hist_comp.plot(hist_orig, color='#d62728', label='Gốc (Đường đỏ)', linewidth=1.5)
        ax_hist_comp.bar(range(256), hist_proc, color='darkblue', width=1, label='Đã xử lý (Cột xanh)', alpha=0.8)
        ax_hist_comp.set_title('So sánh Histogram', fontweight='bold')
        ax_hist_comp.set_ylabel('Số pixel')
        ax_hist_comp.set_xlim([0, 256])
        ax_hist_comp.legend()
        ax_hist_comp.grid(True, linestyle='-', alpha=0.3)

        # 5. Hiển thị Khung Thống kê
        ax_stats.axis('off')  # Ẩn khung viền của ô này
        stats_text = self.generate_stats_text()
        # Đặt chữ vào giữa khung, dùng font monospace để các dấu bằng thẳng hàng
        ax_stats.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', va='center', ha='left')

        # Căn chỉnh lại khoảng cách giữa các biểu đồ
        fig.tight_layout()

    def generate_stats_text(self):
        """Hàm con để trích xuất số liệu thống kê (Min, Max, Mean, Std)"""

        def get_stats(img):
            return {
                'min': np.min(img),
                'max': np.max(img),
                'mean': np.mean(img),
                'std': np.std(img)
            }

        orig = get_stats(self.orig_gray)
        proc = get_stats(self.proc_gray)

        text = "THONG KE HISTOGRAM\n\n"
        text += "Anh goc:\n"
        text += f"- Min:     {orig['min']}\n"
        text += f"- Max:     {orig['max']}\n"
        text += f"- Mean:    {orig['mean']:.2f}\n"
        text += f"- Std Dev: {orig['std']:.2f}\n\n"

        text += "Sau khi xu ly:\n"
        text += f"- Min:     {proc['min']}\n"
        text += f"- Max:     {proc['max']}\n"
        text += f"- Mean:    {proc['mean']:.2f}\n"
        text += f"- Std Dev: {proc['std']:.2f}\n"

        return text


class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # 1. Load file UI
        uic.loadUi('UI_FixIMG.ui', self)

        # 2. Khởi tạo biến
        self.original_img = None
        self.current_img = None
        self.history = []

        # 3. Kết nối Menu Actions
        self.actionOpen.triggered.connect(self.open_image)
        self.actionSave.triggered.connect(self.save_image)

        # 4. Kết nối Sliders
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

        self.btnShowHist.clicked.connect(self.show_histogram)

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

    # --- LOGIC CỐT LÕI ---
    def apply_all_effects(self):
        if self.original_img is None: return

        temp_img = self.original_img.copy()

        # 1. Xử lý Gamma
        gamma_val = self.sliderGamma.value()
        self.lblGammaValue.setText(f"{gamma_val}")

        gamma = gamma_val / 50.0
        if gamma == 0: gamma = 0.01
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        temp_img = cv2.LUT(temp_img, table)

        # 2. Xử lý Khử nhiễu
        denoise_val = self.SliderDenoise.value()
        self.lblDenoiseValue.setText(f"{denoise_val}")

        if denoise_val > 0:
            k_size = (denoise_val // 10) * 2 + 1
            temp_img = cv2.GaussianBlur(temp_img, (k_size, k_size), 0)

        # 3. Xử lý Làm mịn
        smooth_val = self.SliderLammin.value()
        self.lblSmoothValue.setText(f"{smooth_val}")

        if smooth_val > 0:
            d = max(1, smooth_val // 5)
            sigma = smooth_val * 2
            temp_img = cv2.bilateralFilter(temp_img, d, sigma, sigma)

        self.current_img = temp_img
        self.display_image(self.current_img, self.lblProcessed)

    # --- THAO TÁC BIẾN ĐỔI HÌNH HỌC ---
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
        self.sliderGamma.blockSignals(True)
        self.SliderDenoise.blockSignals(True)
        self.SliderLammin.blockSignals(True)

        self.sliderGamma.setValue(50)
        self.SliderDenoise.setValue(0)
        self.SliderLammin.setValue(0)

        self.lblGammaValue.setText("50")
        self.lblDenoiseValue.setText("0")
        self.lblSmoothValue.setText("0")

        self.sliderGamma.blockSignals(False)
        self.SliderDenoise.blockSignals(False)
        self.SliderLammin.blockSignals(False)

    def reset_image(self):
        if self.original_img is not None:
            self.reset_ui_values()
            self.apply_all_effects()

    # --- GỌI CỬA SỔ HISTOGRAM ---
    def show_histogram(self):
        if self.original_img is None or self.current_img is None:
            QMessageBox.warning(self, "Thông báo", "Vui lòng mở ảnh trước khi xem Histogram!")
            return

        dialog = HistogramDialog(self.original_img, self.current_img, self)
        dialog.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ManHinhChao()
    window.show()
    sys.exit(app.exec_())