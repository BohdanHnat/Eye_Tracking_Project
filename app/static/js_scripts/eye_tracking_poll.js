function startEyeTrackingPolling(resultsUrl, statusUrl, totalSeconds) {
    var startTime = Date.now();
    var timerElem = document.getElementById('et-timer-elapsed');
    var hasFinished = false;
    var pollInterval;

    function finish() {
        if (hasFinished) return;
        hasFinished = true;
        if (pollInterval) clearInterval(pollInterval);
        if (timerInterval) clearInterval(timerInterval);
        window.location.href = resultsUrl;
    }

    var timerInterval = setInterval(function() {
        var elapsed = Math.floor((Date.now() - startTime) / 1000);
        if (timerElem) {
            timerElem.textContent = Math.min(elapsed, totalSeconds);
        }
        if (elapsed >= totalSeconds) {
            finish();
        }
    }, 1000);

    pollInterval = setInterval(function() {
        fetch(statusUrl)
            .then(response => response.json())
            .then(data => {
                if (!data.in_progress) {
                    finish();
                }
            })
    }, 2000);
}