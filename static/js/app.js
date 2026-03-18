document.addEventListener("DOMContentLoaded", () => {
    const camSelect = document.getElementById("camera-select");
    const camSelectGroup = document.getElementById("camera-select-group");
    const camInfoGroup = document.getElementById("camera-info-group");
    const camInfo = document.getElementById("camera-info");
    const startStreamBtn = document.getElementById("start-stream-btn");
    const startGameBtn = document.getElementById("start-game-btn");
    const resetGameBtn = document.getElementById("reset-game-btn");
    resetGameBtn.style.display = "none";
    const toast = document.getElementById("toast");

    // Banner elements (above video)
    const bannerIdle = document.getElementById("banner-idle");
    const bannerHud = document.getElementById("banner-hud");
    const bannerGameover = document.getElementById("banner-gameover");
    const bannerPoints = document.getElementById("banner-points");
    const hudRoundsLeft = document.getElementById("hud-rounds-left");
    const hudCountdown = document.getElementById("hud-countdown");
    const hudPoints = document.getElementById("hud-points");

    // Video overlay elements
    const dimOverlay = document.getElementById("video-dim-overlay");
    const flashCorrect = document.getElementById("flash-correct");
    const flashIncorrect = document.getElementById("flash-incorrect");

    // Track streaming state
    let isStreaming = false;

    function showToast(msg, type = "info") {
        toast.textContent = msg;
        toast.className = `toast show is-${type}`;
        setTimeout(() => toast.classList.remove("show"), 3000);
    }

    // ---- Camera UI state helpers ----

    const bannerIdleTitle = document.getElementById("banner-idle-title");

    function setStreamingUI(cameraName) {
        isStreaming = true;
        camSelectGroup.style.display = "none";
        camInfoGroup.style.display = "flex";
        camInfo.textContent = cameraName;
        startStreamBtn.textContent = "Stop Camera";
        startStreamBtn.classList.remove("primary-btn");
        startStreamBtn.classList.add("danger-btn");
        startStreamBtn.disabled = false;
        bannerIdleTitle.textContent = "READY TO PLAY";
        bannerIdleTitle.style.color = "#00d06c";
    }

    function setIdleUI() {
        isStreaming = false;
        camSelectGroup.style.display = "flex";
        camInfoGroup.style.display = "none";
        camInfo.textContent = "";
        startStreamBtn.textContent = "Start Camera";
        startStreamBtn.classList.remove("danger-btn");
        startStreamBtn.classList.add("primary-btn");
        startStreamBtn.disabled = false;
        bannerIdleTitle.textContent = "SELECT CAMERA";
        bannerIdleTitle.style.color = "";
    }

    // ---- Helper to show exactly one banner ----

    function showBanner(active) {
        bannerIdle.style.display = "none";
        bannerHud.style.display = "none";
        bannerGameover.style.display = "none";
        if (active) active.style.display = "flex";
    }

    // ---- Game state polling ----

    function pollGameState() {
        fetch("/api/game_state")
            .then(res => res.json())
            .then(data => updateUI(data))
            .catch(() => {});
    }

    function updateUI(data) {
        const { state, rounds_left, rounds_total, countdown, next_pose, points, correct_pose, incorrect_pose } = data;

        if (state === "idle") {
            showBanner(bannerIdle);
            dimOverlay.classList.add("active");
        }
        else if (state === "playing") {
            showBanner(bannerHud);
            dimOverlay.classList.remove("active");
            hudRoundsLeft.textContent = `Left: ${rounds_left}`;
            hudCountdown.textContent = `${next_pose.toUpperCase()}: ${countdown} s`;
            hudPoints.textContent = `Points: ${points}`;
        }
        else if (state === "game_over") {
            showBanner(bannerGameover);
            dimOverlay.classList.add("active");
            bannerPoints.textContent = `Points: ${points}/${rounds_total}`;
        }

        // Flash feedback
        if (correct_pose) {
            flashCorrect.classList.add("active");
            setTimeout(() => flashCorrect.classList.remove("active"), 400);
        }
        if (incorrect_pose) {
            flashIncorrect.classList.add("active");
            setTimeout(() => flashIncorrect.classList.remove("active"), 400);
        }
        // Toggle game buttons
        if (state === "playing" || state === "game_over") {
            startGameBtn.style.display = "none";
            resetGameBtn.style.display = "";
        } else {
            startGameBtn.style.display = "";
            resetGameBtn.style.display = "none";
        }

    }

    // Poll every 200ms
    setInterval(pollGameState, 200);

    // ---- Camera & game controls ----

    fetch('/api/listCameras')
        .then(res => res.json())
        .then(cameras => {
            cameras.forEach(cam => {
                const opt = document.createElement("option");
                opt.value = "camera:" + cam;
                opt.textContent = cam;
                camSelect.appendChild(opt);
            });
        })
        .catch(e => console.error("Error fetching cameras", e));

    startStreamBtn.addEventListener("click", () => {
        if (!isStreaming) {
            // START CAMERA
            const val = camSelect.value;
            if (!val) return showToast("Select a camera first", "error");

            startStreamBtn.disabled = true;
            showToast("Starting camera...", "info");

            fetch('/api/start_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ stream_src: val })
            }).then(res => res.json()).then(data => {
                if (data.stream) {
                    showToast("Camera started", "success");
                    const img = document.getElementById("video-feed");
                    img.src = "/api/vidFeed?" + new Date().getTime();

                    // Get the display name (text of selected option)
                    const selectedText = camSelect.options[camSelect.selectedIndex].text;
                    setStreamingUI(selectedText);
                } else {
                    showToast(data.message, "error");
                    startStreamBtn.disabled = false;
                }
            }).catch(() => {
                showToast("Failed to reach server", "error");
                startStreamBtn.disabled = false;
            });
        } else {
            // STOP CAMERA
            startStreamBtn.disabled = true;
            showToast("Stopping camera...", "info");

            fetch('/api/stop_stream', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    if (data.ok) {
                        showToast("Camera stopped", "success");
                        document.getElementById("video-feed").src = "";
                        setIdleUI();
                    } else {
                        showToast(data.message, "error");
                        startStreamBtn.disabled = false;
                    }
                }).catch(() => {
                    showToast("Failed to reach server", "error");
                    startStreamBtn.disabled = false;
                });
        }
    });

    // Check if camera is already streaming on page load
fetch('/api/stream_status')
    .then(res => res.json())
    .then(data => {
        if (data.is_streaming && data.stream_src) {
            // Extract camera name from "camera:DEVICE_NAME"
            const camName = data.stream_src.startsWith("camera:")
                ? data.stream_src.substring(7)
                : data.stream_src;
            setStreamingUI(camName);

            const img = document.getElementById("video-feed");
            img.src = "/api/vidFeed?" + new Date().getTime();
        }
    })
    .catch(() => {});

    startGameBtn.addEventListener("click", () => {
        fetch('/api/start_game', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if (data.ok) showToast("Game started!", "success");
                else showToast(data.message, "error");
            }).catch(() => showToast("Error connecting to server", "error"));
    });

    resetGameBtn.addEventListener("click", () => {
        fetch('/api/reset_game', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if (data.ok) showToast("Game reset.", "info");
            }).catch(() => showToast("Error connecting to server", "error"));
    });
});