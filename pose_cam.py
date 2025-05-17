import cv2
import mediapipe as mp
import time
import random
from ultralytics import YOLO  # YOLOv8

# Load YOLO model for person detection
yolo_model = YOLO("yolov8n.pt")  # Load pre-trained YOLOv8 model

# Initialize Mediapipe Pose
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Open webcam
cap = cv2.VideoCapture(0)
pTime = 0

# Define random colors for different persons
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # *Step 1: Detect multiple people using YOLO*
    results = yolo_model(imgRGB)

    # Get detected persons' bounding boxes
    person_boxes = []
    for r in results:
        for box in r.boxes.data:
            x1, y1, x2, y2, score, cls = box.tolist()
            if int(cls) == 0 and score > 0.5:  # Class 0 is 'person'
                person_boxes.append((int(x1), int(y1), int(x2), int(y2)))

    # *Step 2: Apply Pose Estimation for Each Detected Person*
    for i, (x1, y1, x2, y2) in enumerate(person_boxes):
        person_img = imgRGB[y1:y2, x1:x2]  # Crop the detected person
        person_results = pose.process(person_img)

        if person_results.pose_landmarks:
            color = colors[i % len(colors)]  # Assign a different color for each person

            # Draw pose landmarks on the original frame
            for id, lm in enumerate(person_results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * (x2 - x1)) + x1, int(lm.y * (y2 - y1)) + y1
                cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

            mpDraw.draw_landmarks(img, person_results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                                  mpDraw.DrawingSpec(color=color, thickness=2, circle_radius=3),
                                  mpDraw.DrawingSpec(color=color, thickness=2, circle_radius=3))

    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # Show result
    cv2.imshow("Multi-Person Pose Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()