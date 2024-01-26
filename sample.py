import torch
import cv2
import os
import numpy as np
def load_models(directory):
    models = []
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=directory)
            model.eval()
            models.append(model)
    return models

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return torch.tensor(img, dtype=torch.float32)

def run_inference(model, image):
    with torch.no_grad():
        prediction = model(image)
    return prediction

def post_process(prediction):
    # Adapt based on your model's output format
    return classes, scores, boxes

def visualize_results(image, classes, scores, boxes):
    # Adapt based on your visualization needs
    cv2.imshow("YOLOv5 Detection", image)
    cv2.waitKey(1)

weights_directory = "D:/work/facultyDetectWebApp/public/data/weights"
models = load_models(weights_directory)

if not models:
    print("No .pt files found in the specified directory.")
else:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = preprocess_image(frame)

        for model in models:
            prediction = run_inference(model, image)
            classes, scores, boxes = post_process(prediction)
            visualize_results(frame, classes, scores, boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
