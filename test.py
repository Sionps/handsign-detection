import cv2
from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf
import numpy as np
import math
import random
import time

# Load images for display
images = {
    "A": cv2.imread("image/A.png"),
    "B": cv2.imread("image/B.png"),
    "C": cv2.imread("image/C.png"),
    # Add more images if needed
}

# ตรวจสอบการโหลดภาพ
for key, img in images.items():
    if img is None:
        print(f"Error: Image for {key} not found.")
        exit(1)

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

# Load SavedModel and configure prediction signature
model = tf.saved_model.load("Model/model.savedmodel")  # เส้นทางไปยังโมเดล
infer = model.signatures["serving_default"]  # เรียกใช้ signature สำหรับการทำนาย
labels = ["A", "B", "C"]

offset = 20
imgSize = 300
box_x, box_y, box_w, box_h = 300, 100, 300, 300

# เลือกตัวอักษรเป้าหมายแบบสุ่ม
random_letter = random.choice(labels)
top_text = random_letter
score = 0
correct_count = 0

def set_fullscreen(window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def predict(img):
    # เตรียมข้อมูลภาพที่ต้องการทำนาย
    img = np.expand_dims(img, axis=0)  # เพิ่มมิติใหม่
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    # ทำการทำนาย
    prediction = infer(img)
    predicted_class = np.argmax(list(prediction.values())[0].numpy())
    return predicted_class

# Set OpenCV window to full screen
set_fullscreen("Image")

while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # เช็คว่ามืออยู่ในกรอบที่ต้องการ
        if box_x < x < box_x + box_w and box_y < y < box_y + box_h:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # ใช้ imgWhite สำหรับการทำนาย
            predicted_index = predict(imgWhite)
            predicted_label = labels[predicted_index]

            # ตรวจสอบผลลัพธ์การทำนาย
            if predicted_label == top_text:
                correct_count += 1
                if correct_count >= 5:
                    top_text = random.choice(labels)
                    correct_count = 0
                    score += 1
                    if score >= 10:  # รีเซ็ตคะแนนเมื่อถึง 10
                        score = 0

            # แสดงผลลัพธ์
            cv2.putText(imgOutput, predicted_label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)

    # แสดงคะแนนและตัวอักษรเป้าหมาย
    cv2.rectangle(imgOutput, (0, 0), (640, 50), (0, 0, 0), cv2.FILLED)
    cv2.putText(imgOutput, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(imgOutput, top_text, (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # แสดงภาพเป้าหมาย
    if top_text in images:
        letter_img = images[top_text]
        letter_img_resized = cv2.resize(letter_img, (225, 225))
        imgOutput[150:150 + 225, 50:50 + 225] = letter_img_resized

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
