import cv2
import numpy as np
from django.shortcuts import render, redirect
from .forms import ImageUploadForm
# Create your views here.

# Load Yolo
net = cv2.dnn.readNet("/home/tapan/Desktop/code/dev/ObjectTrace/TrackingApp/yolov3.weights",
                      "/home/tapan/Desktop/code/dev/ObjectTrace/TrackingApp/yolov31.cfg")
# save all the names in file o the list classes
classes = []
with open("/home/tapan/Desktop/code/dev/ObjectTrace/TrackingApp/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
# get layers of the network
layer_names = net.getLayerNames()
# Determine the output layer names from the YOLO model
output_layers = [layer_names[i[0] - 1]
                 for i in net.getUnconnectedOutLayers()]


def home(request):
    return render(request, 'home.html')


def detect(request):

 # # Load Yolo
 # net = cv2.dnn.readNet(
 #     "/home/tapan/Desktop/code/dev/ObjectTrace/TrackingApp/yolov3.weights", "/home/tapan/Desktop/code/dev/ObjectTrace/TrackingApp/yolov31.cfg")
 # # save all the names in file o the list classes
 # classes = []
 # with open("/home/tapan/Desktop/code/dev/ObjectTrace/TrackingApp/coco.names", "r") as f:
 #     classes = [line.strip() for line in f.readlines()]
 # # get layers of the network
 # layer_names = net.getLayerNames()
 # # Determine the output layer names from the YOLO model
 # output_layers = [layer_names[i[0] - 1]
 #                  for i in net.getUnconnectedOutLayers()]

    video_capture = cv2.VideoCapture(0)

    counter = 10

    while counter:
        # Capture frame-by-frame
        re, img = video_capture.read()
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # USing blob function of opencv to preprocess image
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        # Detecting objects
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # We use NMS function in opencv to perform Non-maximum Suppression
        # we give it score threshold and nms threshold as arguments.
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(label,	confidences[i])
                cv2.putText(img, text, (x, y - 5), font, 1, color, 1)

        counter -= 1
        cv2.imshow("Image", cv2.resize(img, (800, 600)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

    return redirect('home')


def handle_uploaded_file(f):
    with open('./static/input/image.jpg', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def result(request):

    form = ImageUploadForm(request.POST, request.FILES)

    if form.is_valid():

        handle_uploaded_file(request.FILES['image'])

        img = cv2.imread('./static/input/image.jpg')

        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # USing blob function of opencv to preprocess image
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        # Detecting objects
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # We use NMS function in opencv to perform Non-maximum Suppression
        # we give it score threshold and nms threshold as arguments.
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                text = "{}: {:.2f}".format(label,	confidences[i])
                # cv2.putText(img, label, (x, y + 30), font, 1, color, 3)
                cv2.putText(img, text, (x+3, y+7), font, 0.6, color, 1)

        img = cv2.resize(img, (600, 400))
        cv2.imwrite("./static/output/image.jpg", img)

        return render(request, 'result.html')

    return ""
