// eye_tracking_poll.js
function startEyeTrackingPolling(resultsUrl, statusUrl, totalSeconds) {
    var startTime = Date.now();
    var timerElem = document.getElementById('et-timer-elapsed');
    var timerInterval = setInterval(function() {
        var elapsed = Math.floor((Date.now() - startTime) / 1000);
        if (timerElem) {
            timerElem.textContent = Math.min(elapsed, totalSeconds);
        }
        if (elapsed >= totalSeconds) {
            clearInterval(timerInterval);
        }
    }, 1000);

    // Poll every 2 seconds for completion
    var pollInterval = setInterval(function() {
        fetch(statusUrl)
            .then(response => response.json())
            .then(data => {
                if (!data.in_progress) {
                    clearInterval(pollInterval);
                    clearInterval(timerInterval);
                    window.location.href = resultsUrl;
                }
            });
    }, 2000);
}
