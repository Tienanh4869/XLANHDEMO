import sys
import cv2  # Thư viện xử lý ảnh OpenCV
import numpy as np  # Thư viện tính toán ma trận, mảng
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QDialog, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic  # Dùng để load file giao diện .ui
from PyQt5.QtCore import Qt
from hihi import Ui_Form  # Import giao diện màn hình chào (từ file hihi.py)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# =====================================================================
# LỚP 1: MÀN HÌNH CHÀO (Màn hình đầu tiên xuất hiện khi mở app)
# =====================================================================
class ManHinhChao(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        # Khi bấm nút "Bắt đầu" (pushButton), gọi hàm BatDau
        self.ui.pushButton.clicked.connect(self.BatDau)

    def BatDau(self):
        # Khởi tạo và mở màn hình chính (ImageProcessorApp)
        self.main_app = ImageProcessorApp()
        self.main_app.show()
        # Đóng màn hình chào hiện tại
        self.close()


# =====================================================================
# LỚP 2: CỬA SỔ HIỂN THỊ BIỂU ĐỒ HISTOGRAM VÀ THỐNG KÊ
# =====================================================================
class HistogramDialog(QDialog):
    def __init__(self, orig_img, curr_img, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Phân tích và So sánh Histogram")
        self.resize(1100, 700)

        # Chuyển cả ảnh gốc và ảnh đã xử lý sang ảnh xám (Grayscale) để vẽ biểu đồ
        self.orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        self.proc_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Tạo không gian vẽ biểu đồ (Figure) của Matplotlib
        fig = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)

        # Chia lưới khung vẽ thành 2 hàng, 3 cột
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])

        # Gắn vị trí cho từng thành phần trên lưới
        ax_img_orig = fig.add_subplot(gs[0, 0])  # Ảnh gốc (Góc trên trái)
        ax_hist_orig = fig.add_subplot(gs[0, 1])  # Histogram gốc (Giữa trên)
        ax_img_proc = fig.add_subplot(gs[0, 2])  # Ảnh đã xử lý (Góc trên phải)
        ax_hist_comp = fig.add_subplot(gs[1, 0:2])  # So sánh Histogram (Dưới, chiếm 2 cột)
        ax_stats = fig.add_subplot(gs[1, 2])  # Chữ thống kê (Dưới phải)

        # Tính toán dữ liệu Histogram bằng OpenCV (đếm số pixel cho mỗi mức sáng từ 0-255)
        hist_orig = cv2.calcHist([self.orig_gray], [0], None, [256], [0, 256]).flatten()
        hist_proc = cv2.calcHist([self.proc_gray], [0], None, [256], [0, 256]).flatten()

        # 1. Vẽ ảnh gốc lên khung
        ax_img_orig.imshow(self.orig_gray, cmap='gray')
        ax_img_orig.set_title('Ảnh gốc', fontweight='bold')
        ax_img_orig.axis('off')  # Tắt trục tọa độ

        # 2. Vẽ Histogram ảnh gốc (tô màu đen)
        ax_hist_orig.fill_between(range(256), hist_orig, color='black')
        ax_hist_orig.set_title('Histogram gốc', fontweight='bold')
        ax_hist_orig.set_ylabel('Số pixel')
        ax_hist_orig.set_xlabel('Mức xám')
        ax_hist_orig.set_xlim([0, 256])
        ax_hist_orig.grid(True, linestyle='-', alpha=0.3)

        # 3. Vẽ ảnh sau xử lý
        ax_img_proc.imshow(self.proc_gray, cmap='gray')
        ax_img_proc.set_title('Ảnh sau khi xử lý', fontweight='bold')
        ax_img_proc.axis('off')

        # 4. Vẽ biểu đồ so sánh: Đường đỏ là gốc, Cột xanh là đã xử lý
        ax_hist_comp.plot(hist_orig, color='#d62728', label='Gốc (Đường đỏ)', linewidth=1.5)
        ax_hist_comp.bar(range(256), hist_proc, color='darkblue', width=1, label='Đã xử lý (Cột xanh)', alpha=0.8)
        ax_hist_comp.set_title('So sánh Histogram', fontweight='bold')
        ax_hist_comp.set_ylabel('Số pixel')
        ax_hist_comp.set_xlim([0, 256])
        ax_hist_comp.legend()
        ax_hist_comp.grid(True, linestyle='-', alpha=0.3)

        # 5. In số liệu thống kê bằng chữ
        ax_stats.axis('off')
        stats_text = self.generate_stats_text()
        ax_stats.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', va='center', ha='left')

        fig.tight_layout()  # Căn chỉnh lại khoảng cách cho đẹp

    def generate_stats_text(self):
        """Hàm phụ trợ để tính Min, Max, Mean, Độ lệch chuẩn cho ảnh"""

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


# =====================================================================
# LỚP 3: CỬA SỔ ỨNG DỤNG CHÍNH (Nơi xử lý ảnh)
# =====================================================================
class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Nạp giao diện từ file UI được thiết kế sẵn
        uic.loadUi('UI_FixIMG.ui', self)

        self.source_img = None  # Biến "két sắt": Lưu bản gốc tuyệt đối, không bao giờ bị ghi đè
        self.original_img = None  # Lưu bản nền (có thể bị xoay, cắt, cân bằng...)
        self.current_img = None  # Lưu bản đang được chỉnh sửa hiện tại
        self.history = []  # Mảng lưu lịch sử chỉnh sửa để Undo (Hoàn tác)

        # Kết nối thanh Menu (File -> Mở, Lưu)
        self.actionOpen.triggered.connect(self.open_image)
        self.actionSave.triggered.connect(self.save_image)

        # Kết nối sự kiện khi kéo các thanh trượt (Kéo tới đâu, cập nhật ảnh tới đó)
        self.sliderGamma.valueChanged.connect(self.apply_all_effects)
        self.SliderDenoise.valueChanged.connect(self.apply_all_effects)
        self.SliderLammin.valueChanged.connect(self.apply_all_effects)

        # Khi MỚI BẮT ĐẦU bấm chuột vào thanh trượt -> Lưu lại trạng thái ảnh để lát có thể Undo
        self.sliderGamma.sliderPressed.connect(self.save_state)
        self.SliderDenoise.sliderPressed.connect(self.save_state)
        self.SliderLammin.sliderPressed.connect(self.save_state)

        # Kết nối các nút bấm trên giao diện với các hàm tương ứng
        self.btnHistEq.clicked.connect(self.apply_hist_eq)
        self.btnRotate.clicked.connect(self.rotate_image)
        self.btnFlip.clicked.connect(self.flip_image)
        self.btnCrop.clicked.connect(self.crop_image)
        self.btnUndo.clicked.connect(self.undo_action)
        self.btnReset.clicked.connect(self.reset_image)
        self.btnShowHist.clicked.connect(self.show_histogram)
        self.btnCLAHE.clicked.connect(self.apply_clahe)

    def display_image(self, img, label):
        """Hàm phụ trợ để chuyển đổi ảnh từ OpenCV sang PyQt5 và hiển thị lên lable"""
        if img is None: return
        # OpenCV dùng hệ màu BGR, PyQt5 dùng RGB nên phải chuyển đổi
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        # Tạo đối tượng ảnh của PyQt
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qt_img)
        # Hiển thị lên lable, thu phóng sao cho vừa khung mà không bị méo (KeepAspectRatio)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))

    def open_image(self):
        """Hàm mở file ảnh từ máy tính"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg *.bmp)",
                                                   options=options)

        if file_name:
            # Đọc ảnh vào biến két sắt source_img. Dùng np.fromfile để tránh lỗi khi đường dẫn có tiếng Việt
            self.source_img = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_COLOR)
            if self.source_img is None:
                QMessageBox.warning(self, "Lỗi", "Không thể đọc được file ảnh!")
                return

            self.reset_ui_values()  # Trả các thanh trượt về 0

            # Tạo bản sao từ két sắt ra để xử lý
            self.original_img = self.source_img.copy()
            self.current_img = self.original_img.copy()
            self.history.clear()  # Xóa lịch sử cũ

            # Hiển thị lên 2 khung: Gốc và Đã xử lý
            self.display_image(self.original_img, self.lblOriginal)
            self.display_image(self.current_img, self.lblProcessed)

    def save_image(self):
        """Lưu ảnh hiện hành xuống máy tính"""
        if self.current_img is None: return
        file_name, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh", "", "JPEG (*.jpg);;PNG (*.png)")
        if file_name:
            cv2.imwrite(file_name, self.current_img)
            QMessageBox.information(self, "Thành công", "Đã lưu ảnh!")

    def apply_all_effects(self):
        """Hàm cốt lõi: Kết hợp cả 3 hiệu ứng từ thanh trượt lên ảnh"""
        if self.original_img is None: return

        temp_img = self.original_img.copy()  # Lấy ảnh gốc hiện tại làm nền tảng

        # --- 1. Xử lý sáng tối (Gamma) ---
        gamma_val = self.sliderGamma.value()
        self.lblGammaValue.setText(f"{gamma_val}")  # Hiển thị số lên nhãn
        gamma = gamma_val / 50.0  # Quy đổi để mức 50 (giữa) là 1.0 (không đổi)
        if gamma == 0: gamma = 0.01  # Tránh lỗi chia cho 0
        invGamma = 1.0 / gamma

        # Tạo bảng tra cứu (LookUp Table) để đổi màu pixel nhanh hơn
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        temp_img = cv2.LUT(temp_img, table)

        # --- 2. Khử nhiễu (Denoise) bằng Gaussian Blur ---
        denoise_val = self.SliderDenoise.value()
        self.lblDenoiseValue.setText(f"{denoise_val}")
        if denoise_val > 0:
            k_size = (denoise_val // 10) * 2 + 1  # Đảm bảo kích thước kernel luôn là số lẻ (1, 3, 5...)
            temp_img = cv2.GaussianBlur(temp_img, (k_size, k_size), 0)

        # --- 3. Làm mịn ảnh (Bilateral Filter) ---
        smooth_val = self.SliderLammin.value()
        self.lblSmoothValue.setText(f"{smooth_val}")
        if smooth_val > 0:
            d = max(1, smooth_val // 5)
            sigma = smooth_val * 2
            # Bộ lọc này làm mịn nhưng giữ được các góc cạnh (edges) không bị nhòe
            temp_img = cv2.bilateralFilter(temp_img, d, sigma, sigma)

        # Cập nhật ảnh hiện tại và đưa lên màn hình
        self.current_img = temp_img
        self.display_image(self.current_img, self.lblProcessed)

    # --- CÁC NÚT CÔNG CỤ XỬ LÝ ẢNH CHÍNH ---

    def rotate_image(self):
        if self.original_img is None: return
        self.save_state()  # Lưu trước khi xoay
        # Xoay 90 độ cùng chiều kim đồng hồ
        self.original_img = cv2.rotate(self.original_img, cv2.ROTATE_90_CLOCKWISE)
        self.apply_all_effects()

    def flip_image(self):
        if self.original_img is None: return
        self.save_state()
        # Lật ngang (trái - phải)
        self.original_img = cv2.flip(self.original_img, 1)
        self.apply_all_effects()

    def crop_image(self):
        if self.original_img is None: return
        self.save_state()
        h, w = self.original_img.shape[:2]
        # Cắt đi 10% viền xung quanh, lấy khúc giữa từ 10% đến 90%
        self.original_img = self.original_img[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
        self.apply_all_effects()

    def apply_hist_eq(self):
        """Cân bằng Histogram toàn cục (Dễ gây cháy sáng ở vùng quá sáng/tối)"""
        if self.original_img is None: return
        self.save_state()
        # Không gian YUV: Kênh Y là ánh sáng, U và V là màu sắc
        img_yuv = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # Chỉ cân bằng kênh sáng Y
        self.original_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        self.apply_all_effects()

    def apply_clahe(self):
        if self.original_img is None: return
        self.save_state()

        base_img = self.source_img.copy()  # LUÔN lấy từ két sắt

        img_lab = cv2.cvtColor(base_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))
        self.original_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        self.apply_all_effects()

    # --- CÁC TÍNH NĂNG QUẢN LÝ (Undo, Reset, Show Dialog) ---

    def save_state(self):
        """Lưu lại trạng thái để Hoàn tác (Undo)"""
        if self.original_img is not None:
            # Chỉ lưu tối đa 10 bước để tiết kiệm bộ nhớ (RAM)
            if len(self.history) >= 10:
                self.history.pop(0)

            # Gói gọn ảnh và vị trí các thanh trượt vào 1 Dictionary
            current_state = {
                'image': self.original_img.copy(),
                'gamma': self.sliderGamma.value(),
                'denoise': self.SliderDenoise.value(),
                'smooth': self.SliderLammin.value()
            }
            self.history.append(current_state)

    def undo_action(self):
        """Quay lại bước trước đó"""
        if self.history:
            # Lấy trạng thái cuối cùng ra khỏi danh sách
            previous_state = self.history.pop()
            self.original_img = previous_state['image']

            # Tạm khóa tín hiệu (blockSignals) để lúc set lại số cho thanh trượt
            # nó không tự động kích hoạt hàm apply_all_effects gây lag
            self.sliderGamma.blockSignals(True)
            self.SliderDenoise.blockSignals(True)
            self.SliderLammin.blockSignals(True)

            # Phục hồi giá trị thanh trượt
            self.sliderGamma.setValue(previous_state['gamma'])
            self.SliderDenoise.setValue(previous_state['denoise'])
            self.SliderLammin.setValue(previous_state['smooth'])

            # Phục hồi chữ số hiển thị
            self.lblGammaValue.setText(str(previous_state['gamma']))
            self.lblDenoiseValue.setText(str(previous_state['denoise']))
            self.lblSmoothValue.setText(str(previous_state['smooth']))

            # Mở khóa tín hiệu lại
            self.sliderGamma.blockSignals(False)
            self.SliderDenoise.blockSignals(False)
            self.SliderLammin.blockSignals(False)

            self.apply_all_effects()  # Áp dụng để vẽ lại màn hình

    def reset_ui_values(self):
        """Đưa giao diện thanh trượt về mặc định"""
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
        """Khôi phục ảnh về trạng thái gốc tuyệt đối khi mới mở"""
        if self.source_img is not None:
            self.save_state()  # Lưu trạng thái hiện tại để có thể Undo lệnh Reset

            # Lấy ảnh từ két sắt đè lên ảnh xử lý hiện tại
            self.original_img = self.source_img.copy()

            self.reset_ui_values()
            self.apply_all_effects()

    def show_histogram(self):
        """Mở cửa sổ thống kê Histogram"""
        if self.source_img is None or self.current_img is None:
            QMessageBox.warning(self, "Thông báo", "Vui lòng mở ảnh trước khi xem Histogram!")
            return

        dialog = HistogramDialog(self.source_img, self.current_img, self)
        dialog.exec_()  # Dùng exec_ để mở hộp thoại và khóa tương tác với cửa sổ chính cho đến khi đóng


# Chạy chương trình
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ManHinhChao()
    window.show()
    sys.exit(app.exec_())
