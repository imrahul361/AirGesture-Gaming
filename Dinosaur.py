#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 13:56:15 2020

@author: rahul__10
"""
import tensorflow as tf
import cv2
import multiprocessing as _mp
from Utils import load_graph, detect_hands, predict
from config import RED, GREEN, YELLOW
import keyboard
import numpy as np


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
            if category == "Closed":
                action = 0  # Do nothing
                text = "Run"
                
            elif category == "Open" and y < height/2:
                action = 1 # Jump
                text = "Jump"
                keyboard.press_and_release('up')
            elif category == "Open" and y > height/2:
                action = 2
                text = "Duck"
                keyboard.press_and_release('down')
            else:
                action = 0
                text = "Run"

            with lock:
                v.value = action
            cv2.putText(frame, "{}".format(text), (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, int(height / 2)), YELLOW, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.imshow('Detection', frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
