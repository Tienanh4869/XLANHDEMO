import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from tkinter import Tk, filedialog
from skimage.metrics import structural_similarity as ssim

# --- 1. HỘP THOẠI CHỌN ẢNH ---
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Chọn ảnh thiếu sáng",
                                       filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])

if not file_path:
    print("Bạn chưa chọn ảnh! Chương trình kết thúc.")
    exit()

# Đọc ảnh gốc (Lưu giữ để làm mốc tính toán SSIM/PSNR)
img_bgr = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
img_original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Ảnh dùng để thao tác cộng dồn (HE, CLAHE)
img_base = img_original_rgb.copy()
# Ảnh đang hiển thị hiện tại trên màn hình (Sau khi kéo slider)
img_current_display = img_original_rgb.copy()

# --- 2. THIẾT LẬP CỬA SỔ CHÍNH ---
fig_main, ax_main = plt.subplots(figsize=(10, 8))
fig_main.canvas.manager.set_window_title("Trình Chỉnh Sửa Ảnh - Matplotlib")
plt.subplots_adjust(left=0.1, bottom=0.35)
ax_main.set_title("Trình chỉnh sửa ảnh (Giao diện Matplotlib)", fontweight='bold')
ax_main.axis('off')

im_display = ax_main.imshow(img_base)

# --- 3. TẠO THANH TRƯỢT ---
ax_gamma = plt.axes([0.15, 0.25, 0.7, 0.03], facecolor='lightgray')
ax_denoise = plt.axes([0.15, 0.20, 0.7, 0.03], facecolor='lightgray')
ax_smooth = plt.axes([0.15, 0.15, 0.7, 0.03], facecolor='lightgray')

slider_gamma = Slider(ax_gamma, 'Gamma', 1, 100, valinit=50, valstep=1)
slider_denoise = Slider(ax_denoise, 'Khử nhiễu', 0, 100, valinit=0, valstep=1)
slider_smooth = Slider(ax_smooth, 'Làm mịn', 0, 100, valinit=0, valstep=1)

# --- 4. TẠO NÚT BẤM ---
ax_btn_he = plt.axes([0.1, 0.05, 0.15, 0.05])
ax_btn_clahe = plt.axes([0.28, 0.05, 0.15, 0.05])
ax_btn_report = plt.axes([0.46, 0.05, 0.25, 0.05])  # Nút Report to hơn chút
ax_btn_reset = plt.axes([0.74, 0.05, 0.15, 0.05])

btn_he = Button(ax_btn_he, 'Cân bằng HE', hovercolor='lightblue')
btn_clahe = Button(ax_btn_clahe, 'CLAHE', hovercolor='lightblue')
btn_report = Button(ax_btn_report, 'Xem Histogram & Report', hovercolor='lightgreen')
btn_reset = Button(ax_btn_reset, 'Reset Gốc', hovercolor='salmon')


# --- 5. LOGIC XỬ LÝ ẢNH ---
def update(val=None):
    global img_current_display
    temp_img = img_base.copy()

    gamma_val = slider_gamma.val
    gamma = gamma_val / 50.0
    if gamma == 0: gamma = 0.01
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    temp_img = cv2.LUT(temp_img, table)

    denoise_val = int(slider_denoise.val)
    if denoise_val > 0:
        k_size = (denoise_val // 10) * 2 + 1
        temp_img = cv2.GaussianBlur(temp_img, (k_size, k_size), 0)

    smooth_val = int(slider_smooth.val)
    if smooth_val > 0:
        d = max(1, smooth_val // 5)
        sigma = smooth_val * 2
        temp_img = cv2.bilateralFilter(temp_img, d, sigma, sigma)

    img_current_display = temp_img.copy()  # Lưu lại ảnh hiện tại để xuất report
    im_display.set_data(img_current_display)
    fig_main.canvas.draw_idle()


def apply_he(event):
    global img_base
    img_yuv = cv2.cvtColor(img_base, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_base = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    update()


def apply_clahe(event):
    global img_base
    img_lab = cv2.cvtColor(img_base, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img_lab_merged = cv2.merge((cl, a, b))
    img_base = cv2.cvtColor(img_lab_merged, cv2.COLOR_LAB2RGB)
    update()


def reset_image(event):
    global img_base
    img_base = img_original_rgb.copy()
    slider_gamma.reset()
    slider_denoise.reset()
    slider_smooth.reset()
    update()


# --- 6. LOGIC HIỂN THỊ CỬA SỔ BÁO CÁO (REPORT) ---
def show_report(event):
    # Tạo một cửa sổ Figure mới hoàn toàn độc lập
    fig_hist = plt.figure(figsize=(12, 8))
    fig_hist.canvas.manager.set_window_title("Báo Cáo Phân Tích Histogram")

    orig_gray = cv2.cvtColor(img_original_rgb, cv2.COLOR_RGB2GRAY)
    proc_gray = cv2.cvtColor(img_current_display, cv2.COLOR_RGB2GRAY)

    gs = fig_hist.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])

    ax_img_orig = fig_hist.add_subplot(gs[0, 0])
    ax_hist_orig = fig_hist.add_subplot(gs[0, 1])
    ax_img_proc = fig_hist.add_subplot(gs[0, 2])
    ax_hist_comp = fig_hist.add_subplot(gs[1, 0:2])
    ax_stats = fig_hist.add_subplot(gs[1, 2])

    hist_orig = cv2.calcHist([orig_gray], [0], None, [256], [0, 256]).flatten()
    hist_proc = cv2.calcHist([proc_gray], [0], None, [256], [0, 256]).flatten()

    # Ảnh gốc
    ax_img_orig.imshow(orig_gray, cmap='gray')
    ax_img_orig.set_title('Ảnh gốc', fontweight='bold')
    ax_img_orig.axis('off')

    # Hist gốc
    ax_hist_orig.fill_between(range(256), hist_orig, color='black')
    ax_hist_orig.set_title('Histogram gốc', fontweight='bold')
    ax_hist_orig.set_ylabel('Số pixel')
    ax_hist_orig.set_xlim([0, 256])
    ax_hist_orig.grid(True, linestyle='-', alpha=0.3)

    # Ảnh xử lý
    ax_img_proc.imshow(proc_gray, cmap='gray')
    ax_img_proc.set_title('Ảnh sau khi xử lý', fontweight='bold')
    ax_img_proc.axis('off')

    # So sánh Hist
    ax_hist_comp.plot(hist_orig, color='#d62728', label='Gốc (Đường đỏ)', linewidth=1.5)
    ax_hist_comp.bar(range(256), hist_proc, color='darkblue', width=1, label='Đã xử lý (Cột xanh)', alpha=0.8)
    ax_hist_comp.set_title('So sánh Histogram', fontweight='bold')
    ax_hist_comp.set_xlim([0, 256])
    ax_hist_comp.legend()
    ax_hist_comp.grid(True, linestyle='-', alpha=0.3)

    # Tính toán thông số
    orig_min, orig_max = np.min(orig_gray), np.max(orig_gray)
    orig_mean, orig_std = np.mean(orig_gray), np.std(orig_gray)

    proc_min, proc_max = np.min(proc_gray), np.max(proc_gray)
    proc_mean, proc_std = np.mean(proc_gray), np.std(proc_gray)

    mse_val = np.mean((orig_gray.astype(float) - proc_gray.astype(float)) ** 2)
    psnr_val = float('inf') if mse_val == 0 else 20 * math.log10(255.0 / math.sqrt(mse_val))
    ssim_val = ssim(orig_gray, proc_gray, data_range=255)

    # Tạo Text báo cáo
    text = "THONG KE CO BAN\n"
    text += "-------------------\n"
    text += "Anh goc:\n"
    text += f"- Min:     {orig_min}\n"
    text += f"- Max:     {orig_max}\n"
    text += f"- Mean:    {orig_mean:.2f}\n"
    text += f"- Std Dev: {orig_std:.2f}\n\n"

    text += "Sau khi xu ly:\n"
    text += f"- Min:     {proc_min}\n"
    text += f"- Max:     {proc_max}\n"
    text += f"- Mean:    {proc_mean:.2f}\n"
    text += f"- Std Dev: {proc_std:.2f}\n\n\n"

    text += "DANH GIA CHAT LUONG\n"
    text += "(So voi anh goc)\n"
    text += "-------------------\n"
    text += f"- MSE:     {mse_val:.2f}\n"
    text += f"- PSNR:    {psnr_val:.2f} dB\n"
    text += f"- SSIM:    {ssim_val:.4f}\n"

    ax_stats.axis('off')
    ax_stats.text(0.05, 0.5, text, fontsize=10, family='monospace', va='center', ha='left')

    fig_hist.tight_layout()
    fig_hist.show()  # Hiển thị cửa sổ mới lên


# --- 7. KẾT NỐI SỰ KIỆN ---
slider_gamma.on_changed(update)
slider_denoise.on_changed(update)
slider_smooth.on_changed(update)

btn_he.on_clicked(apply_he)
btn_clahe.on_clicked(apply_clahe)
btn_reset.on_clicked(reset_image)
btn_report.on_clicked(show_report)

# Khởi chạy giao diện chính
plt.show()