import cv2
import numpy as np
import mediapipe as mp
import time
from config import *


# Initialize MediaPipe FaceMesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_styles = mp.solutions.drawing_styles

def detect_primary_movement_direction(normal_vector_prev, normal_vector_current):    
    # 벡터 변화량 계산
    # the type of vector_change is numpy.ndarray.
    # example form : [0.20647509 -0.01026643 0.00863661]
    vector_change = normal_vector_current - normal_vector_prev
    # 각 방향에 대한 변화량 계산
    # vector_change[0] is x coordinates of difference between previous vector and current vector.
    horizontal_change = vector_change[0]
    # vector_change[1] is y coordinates of difference between previous vector and current vector.
    vertical_change = vector_change[1]

    # 변화량이 가장 큰 방향을 결정
    if abs(horizontal_change) > abs(vertical_change):
        # 수평 방향 변화가 더 큼
        if horizontal_change > 0:
            # it seems like right rotation in the screen view.
            # the person is rotating to the left.
            print("right")
            primary_direction = "right"
        else:
            print("left")
            primary_direction = "left"
    else:
        # 수직 방향 변화가 더 큼
        if vertical_change > 0:
            # as like horizontal change, if the person is facing up, the vertical change is less than 0.
            print("up")
            primary_direction = "up"
        else:
            print("down")
            primary_direction = "down"
    
    # 변화량이 없는 경우
    if vector_change[0] == 0 and vector_change[1] == 0:
        primary_direction = "no_change"

    # 사전에서 매핑된 값을 반환
    return direction_mapping[primary_direction]

def draw_normal_vector(image, start_point_2d, normal_vector, scale=50):
    # normal_vector의 z-성분을 무시하고 2D로 표현하기 위해 scale을 조절하여 끝점 계산
    end_point_2d = (int(start_point_2d[0] + normal_vector[0] * scale), 
                    int(start_point_2d[1] - normal_vector[1] * scale))  # y-축은 이미지 좌표계에서 아래로 내려감
    cv2.arrowedLine(image, start_point_2d, end_point_2d, (255, 0, 0), 2, tipLength=0.3)

def draw_vector_lines(image, start_point, end_point, color=(0, 255, 0), thickness=2):
    cv2.line(image, start_point, end_point, color, thickness)

def calculate_angle_change(normal_vector_prev, normal_vector_current):
    unit_vector_prev = normal_vector_prev 
    unit_vector_current = normal_vector_current
    dot_product = np.dot(unit_vector_prev, unit_vector_current)
    # dot product of two unit vector might slightly exceed the range [-1, 1].
    # theses values are out of the domain for the np.arccos.
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)

def detect_headbanging(video_path, verbose=True):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    angle_changes = []
    direction_changes = []
    direction_change_counts = []
    normal_vector_prev = None
    direction_change_count = 0
    last_movement_direction = 0
    last_direction_change_time = time.time()

    x_cord = []
    y_cord = []
    z_cord = []
    count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        # loop as many time as frame count.
        while cap.isOpened():
            # read the next frame. if read successfully, return true to success and return the frame image to image variable.
            success, image = cap.read()
            if not success:
                break
            # get the height and width of the image. _ is number of channel but not going to be used.
            h, w, _ = image.shape
            count += 1
            
            # convert gbr to rgb.
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Process the RGB image through the face mesh model to detect and track facial landmarks.
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
                    # print(len(landmarks)) 478
                    # center_forehead = (landmarks[5] + landmarks[4]) / 2 # 코 중앙
                    left_eye = landmarks[133]
                    right_eye = landmarks[362]

                    # scaling normalised x,y and z cooridnates.
                    left_eye_2d = (int(left_eye[0] * w), int(left_eye[1] * h))
                    right_eye_2d = (int(right_eye[0] * w), int(right_eye[1] * h))

                    cv2.circle(image, left_eye_2d, radius=3, color=(0, 255, 0), thickness=-1)
                    cv2.circle(image, right_eye_2d, radius=3, color=(0, 255, 0), thickness=-1)

                    center_forehead = (left_eye + right_eye) / 2

                    left_lip = landmarks[61]
                    right_lip = landmarks[291]
                    # vector1 points from the 'center_forehead' to the 'left_lip'
                    vector1 = left_lip - center_forehead
                    # vector2 poins from the 'center_forehead' to the 'right_lip'
                    vector2 = right_lip - center_forehead
                    normal_vector = np.cross(vector1, vector2)

                    # 3D coordinates to 2D (ignoring z component)
                    center_forehead_2d = (int(center_forehead[0] * w), int(center_forehead[1] * h))
                    left_lip_2d = (int(left_lip[0] * w), int(left_lip[1] * h))
                    right_lip_2d = (int(right_lip[0] * w), int(right_lip[1] * h))

                    # Draw circles at key landmarks
                    cv2.circle(image, center_forehead_2d, radius=3, color=(0, 255, 0), thickness=-1)
                    cv2.circle(image, left_lip_2d, radius=3, color=(0, 255, 0), thickness=-1)
                    cv2.circle(image, right_lip_2d, radius=3, color=(0, 255, 0), thickness=-1)

                    # Draw vectors from center_forehead to lips
                    draw_vector_lines(image, center_forehead_2d, left_lip_2d)
                    draw_vector_lines(image, center_forehead_2d, right_lip_2d)
                    
                    # if the norm (or magnitude) of the normal_vector is not equal to zero
                    if np.linalg.norm(normal_vector) != 0:
                        # the original normal_vector is divided by it's norm.
                        normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)

                        x_cord.append(normal_vector_normalized[0])
                        y_cord.append(normal_vector_normalized[1])
                        z_cord.append(normal_vector_normalized[2])

                        draw_normal_vector(image, center_forehead_2d, normal_vector_normalized)

                        if normal_vector_prev is not None:
                            # calculate angle between previous vector and current vector and return it as degree form.
                            angle_change = calculate_angle_change(normal_vector_prev, normal_vector_normalized)
                            if angle_change >= 3:
                                current_time = time.time()
                                if current_time - last_direction_change_time > 1:
                                    direction_change_count = 0
                                
                                ''' direction_mapping = {
                                        'no_change': 0,
                                        'up': 1,
                                        'down': -1,
                                        'left': 2,
                                        'right': -2,
                                    }'''
                                # current_movement_direction has direction number.
                                current_movement_direction = detect_primary_movement_direction(normal_vector_prev, normal_vector_normalized)

                                # print(normal_vector_normalized)
                                if last_movement_direction != current_movement_direction: # 좌/우 혹은 상/하
                                    direction_change_count += 1
                                    if verbose : 
                                        print(f"The direction has changed! {direction_change_count}")
                                    last_movement_direction = current_movement_direction
                                    last_direction_change_time = current_time
                                else:
                                    # assume this line is redundant
                                    direction_change_count = direction_change_count
                                # example form of direction_change_counts : [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 1, 1, 2, 2, 2, 2, 3, 0]
                                # if direction_change_count is 3, it means head banging is detected.
                                direction_change_counts.append(direction_change_count)
                                # headbanging_threshold : 3
                                if direction_change_count >= headbanging_threshold:
                                    if verbose : 
                                        current_time_in_seconds = frame_count / fps
                                        minutes = int(current_time_in_seconds // 60)
                                        seconds = int(current_time_in_seconds % 60)
                                        print(f"Headbanging Detected at {minutes}m:{seconds}s")
                                    direction_change_count = 0

                            angle_changes.append(angle_change)
                            
                            # example form of direction_changes : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, -2, -2, -2, -2, -2, -2, 2, 2, 2, 2, 2]
                            # length of direction_changes would be the same as number of frames.
                            direction_changes.append(last_movement_direction)    
                        normal_vector_prev = normal_vector_normalized
                    
                    # 수정된 cv2.circle 호출
                    cv2.circle(image, center_forehead_2d, radius=5, color=(0, 0, 255), thickness=-1)
                    # 랜드마크 시각화 추가
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())

            frame_count += 1
            cv2.imshow('Face Mesh', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    return angle_changes, direction_changes, direction_change_counts, x_cord, y_cord, z_cord
