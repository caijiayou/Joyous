import cv2
import mediapipe as mp

# 初始化 mediapipe 繪圖工具和物體偵測方法
mp_drawing = mp.solutions.drawing_utils  # mediapipe 繪圖方法
mp_objectron = mp.solutions.objectron    # mediapipe 物體偵測方法

cap = cv2.VideoCapture(0)  # 開啟攝影機，0 為預設攝影機

# 啟用物體偵測，偵測鞋子 Shoe
with mp_objectron.Objectron(
    static_image_mode=False,             # 是否為靜態影像模式，設為 False 表示為即時影像
    max_num_objects=5,                  # 設定最大偵測物體數量為 5
    min_detection_confidence=0.5,       # 設定最小偵測信心，0.5 表示50%的偵測信心
    min_tracking_confidence=0.99,       # 設定最小追蹤信心，0.99 表示99%的追蹤信心
    model_name='Shoe') as objectron:    # 設定偵測的物體類型為鞋子

    # 檢查是否能開啟攝影機
    if not cap.isOpened():
        print("Cannot open camera")  # 無法開啟攝影機
        exit()
    
    while True:
        ret, img = cap.read()  # 讀取影像
        if not ret:
            print("Cannot receive frame")  # 若無法獲取影像，則顯示錯誤並退出
            break
        
        img = cv2.resize(img, (520, 300))  # 調整影像大小為 520x300，縮小尺寸加快演算速度
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 顏色格式轉換為 RGB 格式

        results = objectron.process(img2)  # 使用 mediapipe 進行物體偵測

        # 標記所偵測到的物體
        if results.detected_objects:  # 如果偵測到物體
            for detected_object in results.detected_objects:
                # 繪製物體的 2D 特徵點連接
                mp_drawing.draw_landmarks(
                    img, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                # 繪製物體的坐標軸 (旋轉與平移)
                mp_drawing.draw_axis(img, detected_object.rotation, detected_object.translation)

        # 顯示處理後的影像
        cv2.imshow('Joyous Objectron', img)
        
        # 按下 'q' 鍵停止程式
        if cv2.waitKey(5) == ord('q'):
            break  # 按下 'q' 退出

# 釋放攝影機並關閉所有 OpenCV 視窗
cap.release()
cv2.destroyAllWindows()
