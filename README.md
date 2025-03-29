# YOLO + FastSAM 貓咪偵測與遮罩分析工具 🐱

這是一個使用 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 結合 [FastSAM](https://github.com/CASIA-IVA/FastSAM) 的圖像分析腳本。

## 🧠 功能說明

- 使用 YOLO 模型偵測圖片中的貓（class 15）
- 使用 FastSAM 對每隻貓進行遮罩分割
- 儲存結果圖片與分析資料（含面積與輪廓點）
- 所有圖片儲存於 `result_img/`
- 所有分析 CSV 儲存於 `result_csv/`

## 📁 資料夾結構

cat_10fig/ ├── 原始圖片.jpg   ├── result_img/ ← 偵測 + 分割後圖片 
                             └── result_csv/ ← 分析資料 CSV（包含 mask 座標與總表）


## 🚀 執行方式

1. 放入圖片於 `cat_10fig/`
2. 執行 `ultralytic_yolov8_FastSAM.py`
3. 查看輸出圖片與分析結果

## ✅ 環境需求

- Python 3.8+
- `ultralytics`, `opencv-python`, `pandas`, `numpy`

```bash
pip install -r requirements.txt


