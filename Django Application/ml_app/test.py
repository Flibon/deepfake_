import cv2
import threading
import time
import torch
from torch import nn
import numpy as np
from queue import Queue
from torchvision import transforms
from PIL import Image as pImage

# Initialize your model, transformations, and device
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Shared queue to store frames between threads
frame_queue = Queue()
stop_detection = False

# Function to capture frames for 5 seconds, then wait for 20 seconds
def capture_frames():
    cap = cv2.VideoCapture(0)  # Open default camera (use the right index for your camera)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames_to_capture = fps * 5  # Capture frames for 5 seconds
    global stop_detection

    while not stop_detection:
        # Capture frames for 5 seconds
        for _ in range(frames_to_capture):
            ret, frame = cap.read()
            if ret:
                frame_queue.put(frame)  # Add frame to queue for processing
                cv2.imshow('Real-Time Camera', frame)  # Show live camera feed
            else:
                break

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_detection = True
                break

        print("Captured 5 seconds of frames, waiting for 20 seconds...")
        time.sleep(20)  # Wait for 20 seconds before capturing the next set of frames

    cap.release()
    cv2.destroyAllWindows()

# Function to process frames through the model
def process_frames():
    global stop_detection
    while not stop_detection:
        if not frame_queue.empty():
            frames = []
            # Collect all frames from the queue (captured over 5 seconds)
            while not frame_queue.empty():
                frame = frame_queue.get()
                # Apply transformations to each frame
                frame = train_transforms(frame)
                frames.append(frame)

            if frames:
                # Stack frames into a batch and process through the model
                batch = torch.stack(frames).unsqueeze(0).to(device)
                with torch.no_grad():
                    fmap, logits = model(batch)
                    prediction = torch.argmax(logits, dim=1)
                    confidence = sm(logits).max().item() * 100

                    result = "REAL" if prediction.item() == 1 else "FAKE"
                    print(f"Prediction: {result} | Confidence: {confidence:.2f}%")
                
                frames.clear()  # Clear the frame list for the next batch
                time.sleep(20)  # Sleep to match the frame capturing cycle

# Start capturing and processing frames in parallel
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)

capture_thread.start()
process_thread.start()

# Wait for both threads to finish
capture_thread.join()
process_thread.join()

