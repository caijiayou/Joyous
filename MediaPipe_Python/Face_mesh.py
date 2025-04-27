import cv2
import mediapipe as mp

# 載入 mediapipe 的繪圖工具與樣式
mp_drawing = mp.solutions.drawing_utils             # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles     # mediapipe 繪圖樣式
mp_face_mesh = mp.solutions.face_mesh               # mediapipe 人臉網格方法
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)  # 繪圖參數設定（線條厚度和圓點大小）

# 啟動攝影機
cap = cv2.VideoCapture(0)

# 使用 Face Mesh 功能，並設定參數
with mp_face_mesh.FaceMesh(
    max_num_faces=1,                  # 同時偵測最多 1 張人臉
    refine_landmarks=True,            # 提升偵測精度（細緻特徵，如眼睛、嘴巴）
    min_detection_confidence=0.5,     # 偵測信心閾值
    min_tracking_confidence=0.5       # 追蹤信心閾值
) as face_mesh:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    while True:
        ret, img = cap.read()             # 讀取攝影機影像
        if not ret:
            print("Cannot receive frame")
            break
        
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 顏色轉成 RGB（mediapipe 需要）
        results = face_mesh.process(img2)             # 進行人臉網格偵測
        
        if results.multi_face_landmarks:              # 如果有偵測到人臉
            for face_landmarks in results.multi_face_landmarks:
                # 繪製網格 (Tesselation)
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,   # 網格連線
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()
                )
                # 繪製輪廓 (Contours)
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,      # 輪廓線
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style()
                )
                # 繪製虹膜 (Irises)
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,        # 虹膜連線
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style()
                )

        cv2.imshow('Joyous Face_mesh', img)       # 顯示畫面（視窗標題）
        
        if cv2.waitKey(5) == ord('q'):       # 每 5 毫秒檢查鍵盤，按 q 鍵離開
            break

# 釋放攝影機資源並關閉視窗
cap.release()
cv2.destroyAllWindows()
