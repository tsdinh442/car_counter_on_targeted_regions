import cv2
import numpy as np
from predict import *
from ultralytics import YOLO
from utility import masking
from deep_sort.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.detection import Detection

#from shapely.geometry import Polygon

#### draw polygons #####
vertices = []
polygons = []
n = 0


def draw_polygons(event, x, y, flags, params):
    global vertices
    global polygons
    global n
    stroke = 5 # polyline thickness
    yellow = (0, 255, 255)
    radius = stroke
    img = image.copy()

    # check if left click and first point
    if event == cv2.EVENT_LBUTTONDOWN and len(vertices) == 0:
        vertices.append((x, y))
        # draw the first point
        cv2.circle(img, (x, y), radius=radius, color=yellow, thickness=-1)
        cv2.imshow('polygons', img)

    elif event == cv2.EVENT_LBUTTONDOWN and len(vertices) < 2:
        vertices.append((x, y))
        # draw the first line
        cv2.line(img, vertices[0], vertices[-1], color=yellow, thickness=stroke)
        n += 1
        cv2.imshow('polygons', img)

    # check if left_click and the point not close to the first point
    elif event == cv2.EVENT_LBUTTONDOWN and len(vertices) >= 2 and np.linalg.norm(np.array((x, y))
                                                                                 - np.array(vertices[0])) > 30:
        vertices.append((x, y))
        # cv2.circle(img, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
        # draw a line connecting the new point and the previous point
        cv2.polylines(img, np.array([vertices]), isClosed=False, color=yellow, thickness=stroke)
        n += 1
        cv2.imshow('polygons', img)

    # check if the click is the same or close to the first point
    elif event == cv2.EVENT_LBUTTONDOWN and np.linalg.norm(np.array((x, y)) - np.array(vertices[0])) <= 30:
        # close the polyline
        if len(vertices) > 2:
            cv2.polylines(image, np.array([vertices]), isClosed=True, color=(0, 0, 0), thickness=stroke)
            polygons.append(np.array(vertices, dtype=np.int32))
            cv2.imshow('polygons', image)
            vertices = []
            n = 0

    # check if right click
    elif event == cv2.EVENT_RBUTTONDOWN:
        vertices.pop()
        if len(vertices) == 1:
            cv2.circle(img, vertices[0], radius=radius, color=yellow, thickness=-1)
        else:
            cv2.polylines(img, np.array([vertices]), isClosed=False, color=yellow, thickness=stroke)
            n -= 1
        cv2.imshow('polygons', img)


######### main ##############

image_path = '../images/test-1.jpg'
image = cv2.imread(image_path)
image_copy = image.copy()
cv2.imshow('polygons', image)
cv2.setMouseCallback('polygons', draw_polygons)

model = YOLO('../models/11.pt') ##### replace your model here

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == 13:  # Check for Enter key press

        # check if there's a polygon
        if len(polygons) > 0:
            image = masking(image_copy, polygons, opacity=0.5)

            for polygon in polygons:

                # detecting cars and start counting
                count, centers, bboxes, scores = predict(model, polygon, image_path, thresh=0.7, iou=0.7)

                for center in centers:
                    cv2.circle(image, center, radius=10, color=(0, 0, 255), thickness=-1)

                cv2.putText(image, str(count) + 'cars', (polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 2, color=(0, 0, 0), thickness=5)

        cv2.imshow('polygons', image)
        cv2.setMouseCallback('polygons', draw_polygons)

cv2.waitKey(0)
cv2.destroyAllWindows()



