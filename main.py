from ultralytics import YOLO
import cv2
import easyocr
from cam2world_mapper import Cam2WorldMapper
import supervision as sv
import numpy as np
import time
import copy

yolo_model = YOLO('yolo11n.pt')

names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}



# current_frame_vehicle_ids = set()
# current_frame_number_plate_ids = set()

# load video
video_path = './entry_room.webm'

cap = cv2.VideoCapture(video_path)

A, B, C, D = (198, 168), (479, 200), (558, 331), (72, 333)

image_pts = [A, B, C, D]
# M6 is roughly 32 meters wide and 140 meters long there.
world_pts = [(0, 0), (2.5, 0), (3, 2), (0, 2.5)] 

saved_frames = []
person_dict = {}

mapper = Cam2WorldMapper()
mapper.find_perspective_transform(image_pts, world_pts)


######################## Testing on an Image ##############################

# results = plate_detection_model.predict("./car_plate.jpeg")
# img_path = "./car_plate2.jpeg"
# image = cv2.imread(img_path)
# # cv2.imshow("OpenCV Image",image)
# # cv2.waitKey(0)	
# print(image)
# results = plate_detection_model.predict(img_path)
# out = results[0].plot()

# plate = results[0].boxes.xyxy[0]
# x, y, w, h = plate  # Get the bounding box coordinates
# x, y, w, h = int(x), int(y), int(w), int(h)  # Convert to integers

# # Extract the text (if any)
# plate_text = perform_ocr_on_image(image, [x, y, w, h])

#             # Draw the bounding box and the detected text on the frame (optional)
# cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
# cv2.putText(image, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# cv2.imshow('frame', image)
# cv2.waitKey(0)

# print("Detected Plate Text: ", plate_text)

###########################////////////////////////##############################


def generate_compressed_video(saved_frames):
    output_video = "selected_frames_video.mp4"
    frame_rate = 30  # Adjust as needed

        # Read the first frame to get dimensions
    first_frame = saved_frames[0]
    height, width, _ = first_frame.shape

        # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

        # Write selected frames to the video
    for frame in saved_frames:
        out.write(frame)


def detect_entry(person_id : int, transformed_coordinates:list):
    x_t, y_t, w_t, h_t = transformed_coordinates
    if person_id not in person_dict:
        person_dict[person_id] = dict()
        person_dict[person_id]["prev_x_t"] = x_t
        person_dict[person_id]["x_t"] = x_t
        person_dict[person_id]["last_track"] = time.time()

    else:
        if "x_t" in person_dict[person_id]:
            person_dict[person_id]["prev_x_t"] = person_dict[person_id]["x_t"] 
            person_dict[person_id]["x_t"] = x_t
        else:
            person_dict[person_id]["x_t"] = x_t

    
    if abs(person_dict[person_id]["last_track"] - time.time()) >= 1:
        if person_dict[person_id]["x_t"] < person_dict[person_id]["prev_x_t"] and x_t < .5:
            person_dict[person_id]["action"] = "enter"

        elif person_dict[person_id]["x_t"] -.01 > person_dict[person_id]["prev_x_t"] and x_t < .5:
            person_dict[person_id]["action"] = "out"
            print("OUT: ", person_dict[person_id]["x_t"], person_dict[person_id]["prev_x_t"])
        person_dict[person_id]["last_track"] = time.time()



ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:

        results = yolo_model.track(frame, persist=True)
        # print(results[0].boxes)

        for i in range(len(results[0].boxes)):        #### Working on each bounding box element
            box_cls = results[0].boxes.cls[i].tolist()
            if names[box_cls] == "person":
                box_coordinates_video = results[0].boxes.xyxy[i].tolist()
                x, y, w, h = box_coordinates_video
                x, y, w, h = int(x), int(y), int(w), int(h)

                x_t, y_t, w_t, h_t = mapper.map([x, y, w, h]).flatten()

                person_id = results[0].boxes.id[i]
                person_id = int(person_id)

                detect_entry(person_id, [x_t, y_t, w_t, h_t])  ## Detect whether entry or out

                # f.write( str(vehicles))
                # print(vehicles)
                # text = "id: " + str(int(car_id)) + " x: " + str(x) + " y: " + str(y) + " speed: " + str( cars[car_id]["speed"]) + " km/h"
                
                entry_text = " "
                if "action" in person_dict[person_id]:
                    if person_dict[person_id]["action"] == "enter":
                        entry_text = str(names[box_cls]) + " " + "Entering " + " Room"
                    elif person_dict[person_id]["action"] == "out":
                        entry_text = str(names[box_cls]) + " " + "getting OUT of the" + " Room"

                # cv2.rectangle(frame, (220, 5), (330, 40), (0,255,0), 2)
                cv2.putText(frame, entry_text, (120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0, 255), 2)

                cv2.rectangle(frame, (x, y), (w, h), (0,255,0), 2)
                cv2.putText(frame, "Person" + " ", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                if x_t < 0:
                    saved_frames.append(frame)


                    # print(x, y, w, h)

        # print(car_results[0])
        cv2.imshow("cars:", frame)


####################### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #######################

######################## Testing Bounding Regions for Speed Test ###################

        # img = frame
        # # cv2.resize(img, (120, 200))
        # color1 = sv.Color.from_hex("#004080")
        # color2 = sv.Color.from_hex("#f78923")
        # poly = np.array(((198, 168), (479, 200), (558, 331), (72, 333)))  # A=1200, 700 B= 2800, 700 C = 3800, 2200 D= 501, 2200

        # # poly = np.array(((240, 200), (900, 200), (900, 400), (-400, 500)))  # A=1200, 700 B= 2800, 700 C = 3800, 2200 D= 501, 2200


        # img = sv.draw_filled_polygon(img, poly, color1, 0.5)
        # img = sv.draw_polygon(img, poly, sv.Color.WHITE, 12)
        # img = sv.draw_text(img, "A", sv.Point(800, 370), color2, 2, 6)
        # img = sv.draw_text(img, "B", sv.Point(1125, 370), color2, 2, 6)  ## (100, 100), (1200, 100), (1200, 400), (-100, 400)
        # img = sv.draw_text(img, "C", sv.Point(1880, 780), color2, 2, 6)
        # img = sv.draw_text(img, "D", sv.Point(40, 780), color2, 2, 6)

        
        # cv2.imshow("check: ", img)

##################################>>>>>>>>>>################################



        # visualize
        # cv2.imshow('frame', results[0].plot())
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


generate_compressed_video(saved_frames)

