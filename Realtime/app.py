import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import threading
import winsound
import torch.nn.functional as F

# --------------------
# 1️⃣ Device
# --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# 2️⃣ Load Model
# --------------------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 2)
)

model.load_state_dict(torch.load("../notebooks/best_model.pth", map_location=device))
model.to(device)
model.eval()

# --------------------
# 3️⃣ Preprocessing
# --------------------
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------
# 4️⃣ Haar Cascade
# --------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --------------------
# 5️⃣ Buzzer
# --------------------
def beep_alert():
    winsound.Beep(2500, 500)

# --------------------
# 6️⃣ Streamlit UI
# --------------------
st.title("Driver Drowsiness Detection")
run = st.checkbox("Start Camera")
stframe = st.empty()

# --------------------
# 7️⃣ Consecutive Frame Counter
# --------------------
drowsy_counter = 0
buzzer_on = False

# --------------------
# 8️⃣ Camera Loop
# --------------------
if run:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot open camera")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4
            )

            for (x, y, w, h) in faces:

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                face_img = frame[y:y+h, x:x+w]

                input_tensor = preprocess(face_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = F.softmax(output, dim=1)
                    confidence, pred = torch.max(probabilities, dim=1)

                confidence = confidence.item()
                pred = pred.item()

                # Optional confidence threshold
                if confidence < 0.7:
                    pred = 1  # assume AWAKE

                # --------------------------
                # ✅ Consecutive frame logic
                # --------------------------
                if pred == 0:  # Drowsy
                    drowsy_counter += 1
                else:
                    drowsy_counter = 0

                # Only show DROWSY if > 5 consecutive frames
                if drowsy_counter > 5:
                    label = "DROWSY"
                    color = (0, 0, 255)

                    if not buzzer_on:
                        threading.Thread(
                            target=beep_alert,
                            daemon=True
                        ).start()
                        buzzer_on = True
                else:
                    label = "AWAKE"
                    color = (0, 255, 0)
                    buzzer_on = False

                cv2.putText(
                    frame,
                    label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2
                )

            stframe.image(frame, channels="BGR")

        cap.release()
        cv2.destroyAllWindows()
