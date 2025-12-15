from ultralytics import YOLO
import cv2
import requests
import time
from datetime import datetime   

# =========================
# Telegram Configuration
# =========================
# IMPORTANT:
# Replace these with your own Telegram Bot Token and Chat ID
BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"


def send_telegram_alert(image_path, message="⚠️ No Hard Hat Detected!"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    caption = f"{message}\nDetected at: {timestamp}"

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    files = {'photo': open(image_path, 'rb')}
    data = {'chat_id': CHAT_ID, 'caption': caption}
    
    try:
        requests.post(url, files=files, data=data)
        print("Telegram alert sent!")
    except Exception as e:
        print("Error sending Telegram alert:", e)


# =========================
# Load YOLO Model
# =========================
model = YOLO("best.pt")  # place your trained model in the same folder


# =========================
# Camera Setup
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not detected!")
    exit()

alert_cooldown = 3  
last_alert_time = 0


# =========================
# Detection Loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = frame.copy()
    no_hardhat_detected = False

    for r in results:
        for box, cls_id in zip(r.boxes.xyxy, r.boxes.cls):
            cls_name = model.names[int(cls_id)].lower()
            x1, y1, x2, y2 = [int(c) for c in box]

            if "no" in cls_name and "hard" in cls_name:
                no_hardhat_detected = True
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    annotated_frame, cls_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )
            else:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame, cls_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

    if no_hardhat_detected and time.time() - last_alert_time > alert_cooldown:
        img_path = f"alert_{int(time.time())}.jpg"
        cv2.imwrite(img_path, annotated_frame)
        send_telegram_alert(img_path)
        last_alert_time = time.time()

    cv2.imshow("Hardhat Detection + Telegram Alert", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
