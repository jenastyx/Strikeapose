import cv2
import time
import numpy as np
import random
from tensorflow.keras.models import load_model
from utils import (
    initialize_pose_model, process_frame, extract_landmarks, predict_pose_v2,
    blink_screen, draw_bold_text
)

class PoseGameEngine:
    def __init__(self, videoplayer):
        self.videoplayer = videoplayer
        self.pose_model = initialize_pose_model()
        self.label_map = {"X": 0, "Hide": 1, "Pose": 2, "Squat": 3, "Stand": 4}
        self.poses = list(self.label_map.keys())
        self.classifier = load_model("./model/model_strike_a_pose.h5")
        
        # Game State
        self.is_playing = False
        self.is_game_over = False
        self.rounds_total = 10
        self.countdown_total = 5
        self.rounds_left = 10
        self.count = 5
        self.points = 0
        self.next_pose = random.choice(self.poses)
        self.current_pose = None
        self.correct_pose = False
        self.incorrect_pose = False
        self.last_time_check = time.time()
        self.record_time = time.time()
        
        # Overlay settings
        self.rect_top_left = (950, 20)
        self.rect_bottom_right = (350, 700)
        self.rect_color = (0, 255, 0)
        self.rect_thickness = 4

    def update_settings(self, rounds, countdown):
        self.rounds_total = rounds
        self.countdown_total = countdown

    def start_game(self):
        self.is_playing = True
        self.is_game_over = False
        self.rounds_left = self.rounds_total
        self.count = self.countdown_total
        self.points = 0
        self.next_pose = random.choice(self.poses)
        self.current_pose = None
        self.last_time_check = time.time()
        
    def reset_game(self):
        self.is_playing = False
        self.is_game_over = False

    def generate_video_feed(self):
        last_frame_id = 0
        while True:
            if self.videoplayer.streamThread is None or not self.videoplayer.streamThread.is_alive():
                time.sleep(0.1)
                continue

            with self.videoplayer.vid_lock.read_lock():
                current_frame_id = self.videoplayer.frame_id
                frame = self.videoplayer.current_frame

            if current_frame_id <= last_frame_id or frame is None:
                time.sleep(0.005)
                continue
            last_frame_id = current_frame_id

            # frame from videoplayer is RGB. Convert to BGR for cv2 processing/drawing
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Draw ROI rectangle
            cv2.rectangle(frame_bgr, self.rect_top_left, self.rect_bottom_right, self.rect_color, self.rect_thickness)
            
            # process_frame handles drawing landmarks
            results, frame_bgr = process_frame(frame_bgr, self.pose_model)
            frame_bgr = cv2.flip(frame_bgr, 1)

            if self.is_playing and not self.is_game_over:
                if self.rounds_left > 0:
                    if self.correct_pose:
                        frame_bgr, self.correct_pose = blink_screen(frame_bgr, 1, self.record_time, self.correct_pose)
                    elif self.incorrect_pose:
                        frame_bgr, self.incorrect_pose = blink_screen(frame_bgr, 2, self.record_time, self.incorrect_pose)

                    text_playing = [
                        {"text": f"Left: {self.rounds_left}", "position": (50, 100)},
                        {"text": f"{self.next_pose.upper()}: {self.count} s", "position": (430, 100)},
                    ]
                    for data in text_playing:
                        draw_bold_text(frame_bgr, data["text"], data["position"], color=(0, 0, 255))

                    if time.time() - self.last_time_check >= 1:
                        self.count -= 1
                        self.last_time_check = time.time()

                    if self.count == 0:
                        self.count = self.countdown_total
                        self.record_time = time.time()
                        self.current_pose = self.next_pose
                        self.rounds_left -= 1

                        while self.next_pose == self.current_pose:
                            self.next_pose = random.choice(self.poses)

                        landmark_coords = extract_landmarks(results)
                        predicted_pose = predict_pose_v2(landmark_coords, self.label_map, self.classifier)

                        if predicted_pose == self.current_pose:
                            self.correct_pose = True
                            self.points += 1
                        else:
                            self.incorrect_pose = True

                elif self.rounds_left == 0:
                    self.is_game_over = True
                    self.is_playing = False

            if self.is_game_over:
                white_rect = np.zeros_like(frame_bgr) + 255
                alpha = 0.5
                frame_bgr = cv2.addWeighted(frame_bgr, 1 - alpha, white_rect, alpha, 0)
                game_over_text = "GAME OVER!"
                draw_bold_text(frame_bgr, game_over_text, (350, 100), font_scale=3, color=(0, 0, 255), thickness=3)
                points_text = f"Points: {self.points}/{self.rounds_total}"
                draw_bold_text(frame_bgr, points_text, (350, 250), font_scale=3, color=(255, 0, 0), thickness=3)

            if not self.is_playing and not self.is_game_over:
                white_rect = np.zeros_like(frame_bgr) + 255
                alpha = 0.5
                frame_bgr = cv2.addWeighted(frame_bgr, 1 - alpha, white_rect, alpha, 0)
                draw_bold_text(frame_bgr, "READY to play", (400, 350), font_scale=3, color=(0, 0, 255), thickness=4)

            _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
