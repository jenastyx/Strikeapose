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
        self.label_map = {"67": 0, "X": 1, "Hide": 2, "Pose": 3, "Squat": 4, "Stand": 5}
        self.poses = list(self.label_map.keys())
        self.selected_poses = list(self.label_map.keys())
        self.classifier = load_model("./model/model_strike_a_pose.h5")
        
        # Game State
        self.is_playing = False
        self.is_game_over = False
        self.rounds_total = 10
        self.countdown_total = 5
        self.rounds_left = 10
        self.count = 5
        self.points = 0
        self.next_pose = random.choice(self.selected_poses)
        self.current_pose = None
        self.correct_pose = False
        self.incorrect_pose = False
        self.last_time_check = time.time()
        self.record_time = time.time()

        # Visualization settings
        self.show_landmarks = False
        self.show_bbox = True

    def update_settings(self, rounds, countdown):
        self.rounds_total = rounds
        self.countdown_total = countdown

    def update_poses(self, selected_poses):
        self.selected_poses = selected_poses if selected_poses else list(self.label_map.keys())

    def update_visual_settings(self, show_landmarks, show_bbox):
        self.show_landmarks = show_landmarks
        self.show_bbox = show_bbox

    def start_game(self):
        self.is_playing = True
        self.is_game_over = False
        self.rounds_left = self.rounds_total
        self.count = self.countdown_total
        self.points = 0
        self.next_pose = random.choice(self.selected_poses)
        self.current_pose = None
        self.correct_pose = False
        self.incorrect_pose = False
        self.last_time_check = time.time()

    def reset_game(self):
        self.is_playing = False
        self.is_game_over = False
        self.correct_pose = False
        self.incorrect_pose = False

    def get_game_state(self):
        """Return current game state as a dict for the frontend to consume."""
        if self.is_game_over:
            state = "game_over"
        elif self.is_playing:
            state = "playing"
        else:
            state = "idle"

        return {
            "state": state,
            "rounds_left": self.rounds_left,
            "rounds_total": self.rounds_total,
            "countdown": self.count,
            "next_pose": self.next_pose,
            "points": self.points,
            "correct_pose": self.correct_pose,
            "incorrect_pose": self.incorrect_pose,
        }

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

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results, frame_bgr = process_frame(
                frame_bgr, self.pose_model,
                show_landmarks=self.show_landmarks,
                show_bbox=self.show_bbox
            )
            frame_bgr = cv2.flip(frame_bgr, 1)

            if self.is_playing and not self.is_game_over:
                if self.rounds_left > 0:
                    if self.correct_pose:
                        frame_bgr, self.correct_pose = blink_screen(frame_bgr, 1, self.record_time, self.correct_pose)
                    elif self.incorrect_pose:
                        frame_bgr, self.incorrect_pose = blink_screen(frame_bgr, 2, self.record_time, self.incorrect_pose)

                    if time.time() - self.last_time_check >= 1:
                        self.count -= 1
                        self.last_time_check = time.time()

                    if self.count == 0:
                        self.count = self.countdown_total
                        self.record_time = time.time()
                        self.current_pose = self.next_pose
                        self.rounds_left -= 1

                        while self.next_pose == self.current_pose:
                            self.next_pose = random.choice(self.selected_poses)

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
                    self.correct_pose = False
                    self.incorrect_pose = False

            _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")