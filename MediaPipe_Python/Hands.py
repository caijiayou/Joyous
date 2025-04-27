import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_hands = mp.solutions.hands                    # mediapipe 偵測手掌方法

cap = cv2.VideoCapture(0)  # 開啟攝影機

# mediapipe 啟用偵測手掌
with mp_hands.Hands(
    model_complexity=0,  # 模型複雜度設定為0，較簡單的模型
    # max_num_hands=1,   # 限制最多偵測1隻手 (這行被註解掉，可以解註解偵測多隻手)
    min_detection_confidence=0.5,  # 偵測信心度，0到1之間，值越高越精確
    min_tracking_confidence=0.5) as hands:  # 追蹤信心度，0到1之間，值越高越精確

    if not cap.isOpened():  # 檢查攝影機是否開啟成功
        print("Cannot open camera")
        exit()

    while True:
        ret, img = cap.read()  # 讀取每一幀影像
        if not ret:
            print("Cannot receive frame")
            break
        
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 將 BGR 轉換成 RGB 顏色格式，因為 mediapipe 使用的是 RGB
        results = hands.process(img2)  # 偵測手掌

        if results.multi_hand_landmarks:  # 如果偵測到手掌
            for hand_landmarks in results.multi_hand_landmarks:  # 遍歷每一隻手的節點
                # 將節點和骨架繪製到影像中
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,  # 繪製手指間的連線
                    mp_drawing_styles.get_default_hand_landmarks_style(),  # 預設的節點樣式
                    mp_drawing_styles.get_default_hand_connections_style())  # 預設的連線樣式

        cv2.imshow('Joyous Hands', img)  # 顯示畫面
        if cv2.waitKey(5) == ord('q'):  # 按下 'q' 鍵退出循環
            break

cap.release()  # 釋放攝影機
cv2.destroyAllWindows()  # 關閉所有視窗
