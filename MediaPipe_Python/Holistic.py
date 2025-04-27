import cv2
import mediapipe as mp

# 初始化 mediapipe 繪圖工具
mp_drawing = mp.solutions.drawing_utils         # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles # mediapipe 繪圖樣式
mp_holistic = mp.solutions.holistic             # mediapipe 全身偵測方法

cap = cv2.VideoCapture(0)  # 開啟攝影機，0 為預設攝影機

# 使用 mediapipe 啟用全身偵測 (包括面部、身體)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,  # 設定最小偵測信心，0.5 表示50%的偵測信心
    min_tracking_confidence=0.5) as holistic:  # 設定最小追蹤信心，0.5 表示50%的追蹤信心

    # 檢查是否能開啟攝影機
    if not cap.isOpened():
        print("Cannot open camera")  # 無法開啟攝影機
        exit()
    
    while True:
        ret, img = cap.read()  # 讀取影像
        if not ret:
            print("Cannot receive frame")  # 若無法獲取影像，則顯示錯誤並退出
            break
        
        img = cv2.resize(img, (520, 300))  # 調整影像大小為 520x300
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 顏色格式轉換為 RGB 格式

        results = holistic.process(img2)  # 使用 mediapipe 處理影像進行全身偵測

        # 偵測面部，並繪製臉部的網格
        if results.face_landmarks:  # 檢查是否偵測到面部特徵點
            mp_drawing.draw_landmarks(
                img,  # 在原始影像上繪製
                results.face_landmarks,  # 面部特徵點
                mp_holistic.FACEMESH_CONTOURS,  # 面部網格連接點
                landmark_drawing_spec=None,  # 特徵點繪製樣式為預設
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())  # 連接線的繪製樣式
        
        # 偵測身體，並繪製身體骨架
        if results.pose_landmarks:  # 檢查是否偵測到身體骨架
            mp_drawing.draw_landmarks(
                img,  # 在原始影像上繪製
                results.pose_landmarks,  # 身體骨架特徵點
                mp_holistic.POSE_CONNECTIONS,  # 身體各關節連接點
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())  # 身體骨架特徵點的繪製樣式

        # 顯示處理後的影像
        cv2.imshow('Joyous Holistic', img)
        
        # 按下 'q' 鍵停止程式
        if cv2.waitKey(5) == ord('q'):
            break  # 按下 'q' 退出

# 釋放攝影機並關閉所有 OpenCV 視窗
cap.release()
cv2.destroyAllWindows()
