# CLAHE Image Enhancement

Cài đặt **CLAHE** (Contrast Limited Adaptive Histogram Equalization) từ đầu bằng NumPy, so sánh với Histogram Equalization toàn cục và `cv2.createCLAHE` của OpenCV.

![So sánh](results/comparison.png)

## Ý tưởng
Histogram Equalization (HE) toàn cục làm giãn tương phản theo histogram của **cả ảnh**, nên dễ làm cháy vùng sáng và khuếch đại nhiễu. CLAHE khắc phục bằng cách:

1. **Chia ảnh thành lưới tile** (mặc định 8×8) và cân bằng histogram **cục bộ** cho từng tile.
2. **Giới hạn tương phản (clip limit):** cắt các bin histogram vượt ngưỡng và phân phối lại phần dư đều cho mọi bin, tránh khuếch đại nhiễu ở vùng đồng nhất.
3. **Nội suy song tuyến tính** giữa hàm ánh xạ của 4 tile lân cận để không xuất hiện đường biên giữa các tile.

## Cấu trúc
```
clahe.py      # Hàm clahe(img, clip_limit, grid) viết bằng NumPy
demo.py       # So sánh Gốc / HE / CLAHE tự viết / CLAHE OpenCV + histogram + PSNR
results/      # Ảnh kết quả
```

## Chạy
```bash
pip install -r requirements.txt
python demo.py                          # ảnh thử tương phản thấp tự sinh
python demo.py --image anh.jpg --clip 3 --grid 8
```
`demo.py` in ra PSNR giữa bản tự viết và OpenCV để kiểm tra độ chính xác.

## Ứng dụng
Tiền xử lý cho ảnh y tế (X-quang), ảnh chụp thiếu sáng, ảnh kiểm tra quang học (AOI) và làm bước đầu cho các thuật toán phát hiện đặc trưng / thị giác máy tính.
