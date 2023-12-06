import cv2

def predict(model, targeted_regions, image, thresh, iou):

    num_of_cars = 0
    centroids = []
    bboxes = []
    scores = []

    # perform the detection task
    detections = model(image, iou=iou)[0]

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        bboxes.append([x1, y1, x2, y2])
        scores.append(score)
        if score > thresh:
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # test if car inside the marked parking lot region
            if targeted_regions is not None and len(targeted_regions) > 0:
                check = cv2.pointPolygonTest(targeted_regions, centroid, measureDist=False)
            elif not targeted_regions:
                check = True
            if check > 0:
                num_of_cars += 1
                centroids.append(centroid)


    return num_of_cars, centroids, bboxes, scores