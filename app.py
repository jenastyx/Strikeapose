import os
import json
from flask import Flask, Response, render_template, request, jsonify
from flask_cors import CORS
from lib.videoplayer import VideoPlayer
from lib.utils import list_camera_devices, init_logger, log_info
from engine import PoseGameEngine

app = Flask(__name__)
CORS(app)

init_logger()

# 1280x720 video streaming size
videoplayer = VideoPlayer(width=1280, height=720, fps=25)
game_engine = PoseGameEngine(videoplayer)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/settings")
def settings():
    game_engine.reset_game()
    resolution = f"{videoplayer.width}x{videoplayer.height}"
    return render_template("settings.html",
                           rounds=game_engine.rounds_total,
                           countdown=game_engine.countdown_total,
                           resolution=resolution,
                           show_landmarks=game_engine.show_landmarks,
                           show_bbox=game_engine.show_bbox,
                           all_poses=game_engine.poses,
                           selected_poses=game_engine.selected_poses)

@app.route("/api/vidFeed")
def video_feed():
    return Response(game_engine.generate_video_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/game_state")
def game_state():
    return jsonify(game_engine.get_game_state())

@app.route("/api/start_stream", methods=["POST"])
def start_stream():
    payload = request.get_json(silent=True) or {}
    stream_src = payload.get("stream_src")
    if videoplayer.is_started():
        return jsonify({"stream": False, "message": "Stream already started!"})
    videoplayer.start_stream(stream_src)
    return jsonify({"stream": True, "message": "Starting stream..."})

@app.route("/api/stop_stream", methods=["POST"])
def stop_stream():
    if not videoplayer.is_started():
        return jsonify({"ok": False, "message": "No stream running."})
    game_engine.reset_game()
    videoplayer.end_stream()
    return jsonify({"ok": True, "message": "Stream stopped."})

@app.route("/api/start_game", methods=["POST"])
def start_game():
    if not videoplayer.is_started():
         return jsonify({"ok": False, "message": "Start camera first!"})
    game_engine.start_game()
    return jsonify({"ok": True, "message": "Game started!"})

@app.route("/api/reset_game", methods=["POST"])
def reset_game():
    game_engine.reset_game()
    return jsonify({"ok": True})

@app.route("/api/stream_status")
def stream_status():
    return jsonify({
        "is_streaming": videoplayer.is_started(),
        "stream_src": videoplayer._current_stream_src or ""
    })

@app.route("/api/submit_settings", methods=["POST"])
def submit_settings():
    try:
        rounds = int(request.form.get("rounds", 10))
        countdown = int(request.form.get("countdown", 5))
        game_engine.update_settings(rounds, countdown)

        # Handle resolution change
        resolution = request.form.get("resolution", "")
        if resolution and "x" in resolution:
            parts = resolution.split("x")
            new_width = int(parts[0])
            new_height = int(parts[1])

            if new_width != videoplayer.width or new_height != videoplayer.height:
                videoplayer.update_resolution(new_width, new_height)
                log_info(f"Resolution updated to {new_width}x{new_height}")

        # Handle pose selection
        selected_poses = request.form.getlist("poses")
        game_engine.update_poses(selected_poses)

        # Handle visualization toggles
        show_landmarks = request.form.get("show_landmarks") == "on"
        show_bbox = request.form.get("show_bbox") == "on"
        game_engine.update_visual_settings(show_landmarks, show_bbox)
        log_info(f"Visual settings: landmarks={show_landmarks}, bbox={show_bbox}")

        return jsonify({"message": "Settings updated"}), 200
    except Exception as e:
        log_info(f"Failed to update settings: {e}")
        return jsonify({"message": "Failed to update settings"}), 500

@app.route("/api/listCameras", methods=["GET"])
def list_cameras():
    return jsonify(list_camera_devices())



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1336, debug=True, use_reloader=False)