import cv2
import mediapipe as mp


# 開啟攝影機
cap = cv2.VideoCapture(0)

# 載入 Mediapipe 的人臉偵測與繪圖工具
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 使用 FaceDetection，設定 model_selection=0，最小偵測信心值為 0.5
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
    
    if not cap.isOpened():
        print("無法開啟攝影機")
        exit()
    
    while True:
        ret, img = cap.read()  # 讀取一幀影像
        if not ret:
            print("無法讀取畫面")
            break
    
        img.flags.writeable = False  # 設定影像為唯讀，加速處理
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換成 RGB 色彩 (Mediapipe 使用 RGB)
        results = face_detection.process(img)  # 進行人臉偵測
    
        img.flags.writeable = True  # 允許影像寫入
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 再轉回 BGR，方便 OpenCV 顯示
    
        if results.detections:
            print(len(results.detections))  # 顯示偵測到的人臉數量
            for detection in results.detections:
                mp_drawing.draw_detection(img, detection)  # 在人臉上畫框與標記
    
        cv2.imshow('Joyous Face_Detection', img)  # 顯示影像
        if cv2.waitKey(1) == ord('q'):
            break  # 按下 q 鍵停止程式

# 釋放資源
cap.release()
cv2.destroyAllWindows()
