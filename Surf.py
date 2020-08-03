#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 16:29:18 2020

@author: rahul__10
"""

import tensorflow as tf
import cv2
import multiprocessing as _mp
from Utils import load_graph, detect_hands, predict,is_in_triangle
from config import RED, CYAN, YELLOW, GREEN, MAGENTA, BLUE
import keyboard
import numpy as np
import math 

width = 640
height = 480
threshold = 0.6
alpha = 0.3
pre_trained_model_path = "/Users/rahul__10/Desktop/html/HANDGESTUREMODEL/models/pretrained_model.pb" 


def main():
    graph, sess = load_graph(pre_trained_model_path)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    mp = _mp.get_context("spawn")
    v = mp.Value('i', 0)
    lock = mp.Lock()
    
    
    x_center = int(width/2)
    y_center = int(height/2)
    radius = int(min(width,height)/6)
    
    while True:
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detect_hands(frame, graph, sess)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = predict(boxes, scores, classes, threshold, width, height)

        if len(results) == 1:
            x_min, x_max, y_min, y_max, category = results[0]
            x = int((x_min + x_max) / 2)
            y = int((y_min + y_max) / 2)
            cv2.circle(frame, (x, y), 5, RED, -1)
            if category == "Open" and math.sqrt((x-x_center)**2+(y-y_center)**2) <= radius:
                action = 0  # Do nothing
                text = "Stay"
                
            elif category == "Closed" and is_in_triangle((x,y), [(0,0),(width,0),(x_center,y_center)]):
                action = 1 # Jump
                text = "Jump"
                keyboard.press_and_release("up")
            elif category == "Closed" and is_in_triangle((x,y), [(0,height),(width,height),(x_center,y_center)]):
                action = 2 # Down
                text = "Duck"
                keyboard.press_and_release("down")
            elif category == "Closed" and is_in_triangle((x,y), [(0,0),(0,height),(x_center,y_center)]):
                action = 3 # Left
                text = "Left"
                keyboard.press_and_release("left")
            elif category == "Closed" and is_in_triangle((x,y), [(width,0),(width,height),(x_center,y_center)]):
                action = 4 # Right
                text = "Right"
                keyboard.press_and_release("right")
            else:
                action = 0 # Do nothing
                text = "Stay"

            with lock:
                v.value = action
            cv2.putText(frame, "{}".format(text), (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
        overlay = frame.copy()
        cv2.drawContours(overlay,[np.array([(0,0),(width,0),(x_center,y_center)])],0,MAGENTA,-1)   
        cv2.drawContours(overlay,[np.array([(0,height),(width,height),(x_center,y_center)])],0,MAGENTA,-1)  
        cv2.drawContours(overlay,[np.array([(0,0),(0,height),(x_center,y_center)])],0,BLUE,-1)  
        cv2.drawContours(overlay,[np.array([(width,0),(width,height),(x_center,y_center)])],0,BLUE,-1)  
        cv2.circle(overlay, (x_center, y_center),radius , YELLOW, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.imshow('Detection', frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
