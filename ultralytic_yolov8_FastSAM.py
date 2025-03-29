import os
import numpy as np
import cv2
import time
import pandas as pd
from ultralytics import YOLO, FastSAM

def analyze_bboxes(boxes, result_filename):
    data = []
    for i, xywh_tensor in enumerate(boxes.xywh):
        x, y, w, h = xywh_tensor.tolist()
        bbox = boxes.xyxy[i]
        x_min, y_min, x_max, y_max = bbox.tolist()
        area = (x_max - x_min) * (y_max - y_min)
        data.append({
            "NAME": result_filename,
            "TYPE": "BBox",
            "ID": i,
            "Center_X": round(x, 2),
            "Center_Y": round(y, 2),
            "Width": round(w, 2),
            "Height": round(h, 2),
            "Area": round(area, 2)
        })
    return data
def analyze_masks(masks, result_filename):# 
    data = []
    for j, contour in enumerate(masks.xy):
        x = contour[:, 0]
        y = contour[:, 1]
        polygon_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        data.append({
            "NAME": result_filename,
            "TYPE": "Mask",
            "ID": j,
            "Point_Count": len(contour),
            "Area": round(polygon_area, 2)
        })
    return data
def save_analysis_to_csv(bbox_data, mask_data, output_csv):     #✅ 最後儲存成 CSV
    all_data = bbox_data + mask_data
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 分析結果已儲存至：{output_csv}")
def analyze_masks_and_save_points(masks, result_filename, output_folder):
    summary_data = []
    point_data = []

    for j, contour in enumerate(masks.xy):
        x = contour[:, 0]
        y = contour[:, 1]
        polygon_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        # 儲存點資料
        for idx, (xi, yi) in enumerate(zip(x, y)):
            point_data.append({
                "Mask_ID": j,
                "Point_ID": idx + 1,
                "X": int(xi),
                "Y": int(yi)
            })

        # 遮罩總結資訊
        summary_data.append({
            "NAME": result_filename,
            "TYPE": "Mask",
            "ID": j,
            "Point_Count": len(contour),
            "Area": round(polygon_area, 2)
        })

    # 儲存點資料成 CSV
    point_df = pd.DataFrame(point_data)
    csv_path = os.path.join(output_folder, f"{os.path.splitext(result_filename)[0]}_mask_points.csv")
    point_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"🌀 遮罩輪廓點資料已儲存：{csv_path}")

    return summary_data


start_time = time.time()


# 載入模型
yolo_model = YOLO("yolo11m.pt")         # 或 yolov8n.pt/yolov8s.pt，視需求而定
fastsam_model = FastSAM("FastSAM-x.pt") # 或 FastSAM-s.pt

# 圖片資料夾
image_folder = "cat_10fig"

for filename in os.listdir(image_folder):
        # 建立 Img 與 CSV 資料夾
    img_dir = os.path.join(image_folder, "result_img")
    csv_dir = os.path.join(image_folder, "result_csv")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(image_folder, filename)
        print(f"\n📷 處理圖片：{image_path}")

        # 1️⃣ YOLO 偵測貓（類別 ID 15 是貓）
        yolo_results = yolo_model(image_path, device=0)[0]
        cat_bboxes = []
        for box, cls in zip(yolo_results.boxes.xyxy, yolo_results.boxes.cls):
            if int(cls) == 15:  # 類別 15 是 cat
                cat_bboxes.append(box.tolist())

        print(f"🔍 偵測到 {len(cat_bboxes)} 隻貓")

        # 2️⃣ 用 FastSAM 對每個 bbox 做分割
        for idx, bbox in enumerate(cat_bboxes):
            x1, y1, x2, y2 = [int(x) for x in bbox]
            results = fastsam_model(image_path, bboxes=[x1, y1, x2, y2], retina_masks=True)

            for r_i, result in enumerate(results):
                print(f"\n== 第 {idx+1} 隻貓 - FastSAM 分割結果 ==")
                # 建立輸出結果檔名
                result_filename = f"result_{os.path.splitext(filename)[0]}_{idx+1}.jpg"

                # 分析結果（有回傳值！）
                bbox_result = analyze_bboxes(result.boxes, result_filename)
                mask_result = analyze_masks(result.masks, result_filename)
                # 儲存分析結果（append 模式避免每張圖覆蓋）
                csv_path = os.path.join(csv_dir, "analysis_result.csv")
                if not os.path.exists(csv_path):
                    save_analysis_to_csv(bbox_result, mask_result, csv_path)  # 第一次建立
                    
                else:
                    all_data = bbox_result + mask_result
                    df = pd.DataFrame(all_data)
                    df.to_csv(csv_path, mode='a', index=False, header=False, encoding='utf-8-sig')  # 後續追加
                    print(f"✅ 分析追加儲存至：{csv_path}")
                
                analyze_masks_and_save_points(result.masks, result_filename, csv_dir)  # 儲存點資料

                # 儲存到 cat_fig 資料夾中
                result_path = os.path.join(img_dir, result_filename)
                result.save(filename=result_path)
                print(f"✅ 結果儲存於：{result_path}")

                # 讀取圖片以顯示
                cv2.imread(result_path)  
                cv2.imshow("Result", cv2.imread(result_path))
                cv2.waitKey(1)  # 等待按鍵事件
                cv2.destroyAllWindows()
                print(f"🔄 圖片 {result_filename} 已儲存。")

# 過濾出只有檔案的項目（排除資料夾），也可以加上副檔名過濾
image_files = [
    f for f in os.listdir(img_dir)
    if os.path.isfile(os.path.join(img_dir, f)) and f.lower().endswith('.jpg')
]

# 計算檔案數量
file_count = len(image_files)
print(f"目前 result_img 資料夾中有 {file_count} 個 .jpg 檔案")   

end_time = time.time()
elapsed_time = end_time - start_time
print(f"⏱️ 總處理時間：{elapsed_time:.2f} 秒")
# 釋放資源
cv2.destroyAllWindows()
# 釋放所有資源
