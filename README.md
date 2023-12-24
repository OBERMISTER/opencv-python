# opencv-python
opencv-python
import cv2
import numpy as np

def detect_objects(image_path, confidence_threshold=0.5, nms_threshold=0.4):
    # Load YOLO model with COCO dataset classes
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    classes = []
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Load image and get image details
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Create a blob from the image and pass it through the network
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Initialize lists to store bounding boxes, confidences, and class IDs
    boxes, confidences, class_ids = [], [], []

    # Process each output layer
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x, y = int(center_x - w/2), int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to eliminate overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Draw bounding boxes and labels on the image
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = colors[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the result
    cv2.imshow('Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'path/to/your/image.jpg'
detect_objects(image_path)
