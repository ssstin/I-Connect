from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tempfile
import os
import cv2 as cv
import mediapipe as mp
import copy
import numpy as np
import itertools
from collections import Counter, deque
import base64
import json
import time
import csv
import asyncio

from speechtospeech.can_to_eng import Can_to_eng
from speechtospeech.eng_to_can import Eng_to_can
from cv.model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from cv.model.point_history_classifier.point_history_classifier import PointHistoryClassifier

app = FastAPI()

# Only mount static directory if it exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """ Serves the HTML page for the web interface """
    return templates.TemplateResponse("index.html", {"request": request})
    
async def process_audio(websocket: WebSocket, translator_class):
    await websocket.accept()
    print(f"✅ Client connected for {translator_class.__name__} translation...")

    while True:
        try:
            print("🎤 Waiting for audio data...")
            audio_data = await websocket.receive_bytes()
            print(f"✅ Received audio data: {len(audio_data)} bytes")

            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                temp_wav.write(audio_data)
                audio_path = temp_wav.name

            print(f"✅ Audio saved as {audio_path}")

            # Perform Translation
            transcription, translation = translator_class.translate_audio(audio_path)

            # Send response back to UI
            await websocket.send_json({"original": transcription, "translated": translation})

            # Delete file after processing
            os.remove(audio_path)

        except Exception as e:
            print(f"❌ Error: {e}")
            await websocket.close()
            break

@app.websocket("/can_to_eng")
async def cantonese_to_english(websocket: WebSocket):
    await process_audio(websocket, Can_to_eng)

@app.websocket("/eng_to_can")
async def english_to_cantonese(websocket: WebSocket):
    await process_audio(websocket, Eng_to_can)

@app.websocket("/hand_gesture")
async def hand_gesture_recognition(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected for hand gesture recognition...")

    # Initialize MediaPipe Hands - with minimal complexity for Raspberry Pi
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # Limit to one hand for performance
        min_detection_confidence=0.5,  # Lower threshold for better performance
        min_tracking_confidence=0.5,
        model_complexity=0  # Use the lightest model for Raspberry Pi
    )
    
    # Initialize the classifiers
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()
    
    # Read label files
    keypoint_classifier_labels = []
    point_history_classifier_labels = []
    try:
        with open("cv/model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig") as f:
            keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
        with open("cv/model/point_history_classifier/point_history_classifier_label.csv", encoding="utf-8-sig") as f:
            point_history_classifier_labels = [row[0] for row in csv.reader(f)]
        print("✅ Successfully loaded label files")
    except FileNotFoundError as e:
        print(f"❌ Label files not found: {str(e)}")
        keypoint_classifier_labels = ["Open", "Close", "Point", "OK"]
        point_history_classifier_labels = ["None", "Clockwise", "Counter Clockwise"]
    
    # Initialize variables for tracking
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    
    # Frame skipping for performance
    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 3  # Only process every 3rd frame
    
    while True:
        try:
            # Receive the frame from client
            data = await websocket.receive_text()
            data = json.loads(data)
            image_data = data.get("image")
            
            if not image_data:
                continue
            
            # Frame skipping for performance
            frame_count += 1
            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                # Skip this frame processing, just acknowledge receipt
                await websocket.send_json({"status": "skipped"})
                continue
                
            # Decode base64 image - reduced resolution for performance
            encoded_data = image_data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            image = cv.imdecode(nparr, cv.IMREAD_COLOR)
            
            # Resize image for faster processing
            height, width = image.shape[:2]
            scale_factor = 0.5  # Reduce size by half for Raspberry Pi
            image = cv.resize(image, (int(width * scale_factor), int(height * scale_factor)))
            
            # Add this line to flip the image horizontally (mirror it):
            image = cv.flip(image, 1)  # 1 means horizontal flip

            # Process image for hand detection
            debug_image = copy.deepcopy(image)
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True
            
            detected_gesture = {
                "hand_sign": "",
                "finger_gesture": ""
            }
            
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Calculate bounding rectangle
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    
                    # Calculate landmark coordinates
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                    
                    # Classify hand gesture
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    
                    # Update point history
                    if hand_sign_id == 2:  # Point gesture
                        point_history.append(landmark_list[8])  # Index fingertip
                    else:
                        point_history.append([0, 0])
                    
                    # Classify finger gesture
                    finger_gesture_id = 0
                    if len(pre_processed_point_history_list) == (history_length * 2):
                        finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                    
                    # Add to gesture history
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(finger_gesture_history).most_common()
                    most_common_gesture_id = most_common_fg_id[0][0] if most_common_fg_id else 0
                    
                    # Simplified visualization for Raspberry Pi
                    # Just draw essential elements to save processing power
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_bounding_rect(True, debug_image, brect)
                    
                    # Add text info to image (minimized for performance)
                    cv.putText(debug_image, 
                              f"{handedness.classification[0].label[0]}: {keypoint_classifier_labels[hand_sign_id]}", 
                              (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)
                    
                    # Store detected gesture
                    detected_gesture["hand_sign"] = keypoint_classifier_labels[hand_sign_id]
                    detected_gesture["finger_gesture"] = point_history_classifier_labels[most_common_gesture_id]
            else:
                point_history.append([0, 0])
            
            # Draw point history (simplified)
            for index, point in enumerate(point_history):
                if point[0] != 0 and point[1] != 0:
                    cv.circle(debug_image, (point[0], point[1]), 1 + index//2, (152, 251, 152), 2)
            
            # Convert processed image back to base64 for sending to client
            # Further reduce image quality for network performance on Raspberry Pi
            encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 70]  # Lower quality for faster transmission
            _, buffer = cv.imencode('.jpg', debug_image, encode_param)
            output_image = base64.b64encode(buffer).decode('utf-8')
            
            # Send results back to client
            await websocket.send_json({
                "processed_image": f"data:image/jpeg;base64,{output_image}",
                "gesture": detected_gesture
            })
            
            # Add a small delay to prevent overwhelming the Raspberry Pi
            await asyncio.sleep(0.05)
            
        except Exception as e:
            print(f"❌ Error in hand gesture processing: {str(e)}")
            import traceback
            traceback.print_exc()
            await websocket.close()
            break
    
    hands.close()

# Helper functions for hand gesture recognition - simplified for Raspberry Pi
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list))) if temp_landmark_list else 1
    def normalize_(n):
        return n / max_value if max_value != 0 else 0
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width if image_width != 0 else 0
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height if image_height != 0 else 0
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    return temp_point_history

def draw_landmarks(image, landmark_point):
    # Simplified drawing for performance
    connections = [
        (2, 3), (3, 4),  # Thumb
        (5, 6), (6, 7), (7, 8),  # Index finger
        (9, 10), (10, 11), (11, 12),  # Middle finger
        (13, 14), (14, 15), (15, 16),  # Ring finger
        (17, 18), (18, 19), (19, 20),  # Little finger
        (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)  # Palm
    ]
    
    # Draw only the connections for performance
    for connection in connections:
        if len(landmark_point) > max(connection):
            cv.line(image, tuple(landmark_point[connection[0]]), 
                    tuple(landmark_point[connection[1]]), (255, 255, 255), 1)
    
    # Draw only the fingertips for performance
    fingertips = [4, 8, 12, 16, 20]
    for index in fingertips:
        if len(landmark_point) > index:
            cv.circle(image, (landmark_point[index][0], landmark_point[index][1]), 4, (255, 255, 255), -1)
            
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)