document.addEventListener("DOMContentLoaded", () => {
    const camSelect = document.getElementById("camera-select");
    const startStreamBtn = document.getElementById("start-stream-btn");
    const startGameBtn = document.getElementById("start-game-btn");
    const resetGameBtn = document.getElementById("reset-game-btn");
    const toast = document.getElementById("toast");

    function showToast(msg, type = "info") {
        toast.textContent = msg;
        toast.className = `toast show is-${type}`;
        setTimeout(() => toast.classList.remove("show"), 3000);
    }

    // Fetch cameras
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
                // Refresh the video feed image source to trigger reload of the multipart stream
                const img = document.getElementById("video-feed");
                img.src = "/api/vidFeed?" + new Date().getTime();
            } else {
                showToast(data.message, "error");
                startStreamBtn.disabled = false;
            }
        }).catch(err => {
            showToast("Failed to reach server", "error");
            startStreamBtn.disabled = false;
        });
    });

    startGameBtn.addEventListener("click", () => {
        fetch('/api/start_game', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if (data.ok) showToast("Game started!", "success");
                else showToast(data.message, "error");
            }).catch(e => showToast("Error connecting to server", "error"));
    });

    resetGameBtn.addEventListener("click", () => {
        fetch('/api/reset_game', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if (data.ok) showToast("Game reset.", "info");
            }).catch(e => showToast("Error connecting to server", "error"));
    });
});
