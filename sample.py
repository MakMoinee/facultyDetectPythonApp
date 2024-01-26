import torch
import cv2
import os
import numpy as np
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from datetime import datetime
import mysql.connector


acceptable_confidence = 0.5
detectedCount = 0

mydb = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="Develop@2021",
    database="facultydetect"
)
mycursor = mydb.cursor()

def save_detection(image_name, className):
    print("Saving detection ...")
    detectionLogs = {}
    imagePath = f"./results/{image_name}" 
    det = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detectionLogs['status'] = "Unread"
    sql = "INSERT INTO results (facultyID, name, detectedDate, imagePath,status) VALUES (%s,%s, %s, %s, %s)"
    val = (1,className.lower(),det, imagePath, 'faculty detected')
    mycursor.execute(sql, val)
    print("Successfully saved detection")
    mydb.commit()
    
    
def save_image_with_boxes(frame, detections):
    detected_objects = []
    for index, detection in detections.iterrows():
        if detection['confidence'] >= acceptable_confidence:
            # box = [
            #     int(detection['xmin']),
            #     int(detection['ymin']),
            #     int(detection['xmax']),
            #     int(detection['ymax'])
            # ]
            # # Draw bounding box on the frame
            # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            # cv2.putText(frame, f"{detection['name']} {detection['confidence']:.2f}",
            #             (box[0], box[1] - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            detected_objects.append({
                'name': detection['name'],
                'confidence': detection['confidence']
            })

    if detected_objects:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_name = f"detected_{timestamp}.jpg"
        cv2.imwrite(f"D:/work/facultyDetectWebApp/public/results/{image_name}", frame)
        return image_name, detected_objects

    return None, None

def load_models(directory):
    models = []
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            model_path = os.path.join(directory, filename)
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            model.eval()
            models.append(model)
    return models

def preprocess_image(image):
    if isinstance(image, str):
        img = cv2.imread(image)
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise ValueError("Invalid image type")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return torch.tensor(img, dtype=torch.float32)

def run_inference(model, image):
    with torch.no_grad():
        prediction = model(image)
        # detections = prediction.pandas().xyxy[0]
        
        # for index, detection in detections.iterrows():
        #     if (detection['confidence'] >= acceptable_confidence):
        #         print(f"Confidence: {detection['confidence']}, Name: {detection['name']}")
    return prediction

def post_process(prediction):
    # Replace with actual post-processing logic based on your model's output format
    classes = None
    scores = None
    boxes = None
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

        # image = preprocess_image(frame)

        for model in models:
            results = run_inference(model, frame)
            frame = results.render()[0]

            detections = results.pandas().xyxy[0]
        
            for index, detection in detections.iterrows():
                if (detection['confidence'] >= acceptable_confidence):
                    print(f"Confidence: {detection['confidence']}, Name: {detection['name']}")
                    detectedCount +=1
                    if(detectedCount==10):
                        detectedCount = 0
                        im,s = save_image_with_boxes(frame,detections)
                        save_detection(im,detection['name'])

        # Display the resulting frame
        cv2.imshow('YOLOv5 Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    

    cap.release()
    cv2.destroyAllWindows()
