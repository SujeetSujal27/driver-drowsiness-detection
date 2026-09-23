// ========================================
// MEDIAPIPE
// ========================================

import {
    FaceLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs";


// ========================================
// HTML ELEMENTS
// ========================================

const startBtn =
    document.getElementById("startBtn");

const stopBtn =
    document.getElementById("stopBtn");

const camera =
    document.getElementById("camera");

const cameraPlaceholder =
    document.getElementById("cameraPlaceholder");

const timer =
    document.getElementById("timer");

const systemStatus =
    document.getElementById("systemStatus");

const headerStatus =
    document.getElementById("headerStatus");

const cameraStatus =
    document.querySelector(".camera-status");

const faceStatus =
    document.getElementById("faceStatus");

const eyeDetection =
    document.getElementById("eyeDetection");

const eyeStatus =
    document.getElementById("eyeStatus");

const driverStatus =
    document.getElementById("driverStatus");

const statusMessage =
    document.getElementById("statusMessage");

const drowsinessLevel =
    document.getElementById("drowsinessLevel");

const alarmStatus =
    document.getElementById("alarmStatus");

const driverStatusBox =
    document.querySelector(".driver-status");

const statusIcon =
    document.querySelector(".status-icon");


// ========================================
// STATISTICS
// ========================================

const drowsinessEvents =
    document.getElementById("drowsinessEvents");

const alarmActivations =
    document.getElementById("alarmActivations");

const eyesClosedTime =
    document.getElementById("eyesClosedTime");

const currentStatus =
    document.getElementById("currentStatus");


// ========================================
// AI ANALYSIS
// ========================================

const earValue =
    document.getElementById("earValue");

const earState =
    document.getElementById("earState");

const earGraph =
    document.getElementById("earGraph");

const graphContext =
    earGraph.getContext("2d");


// ========================================
// EVENT LOG
// ========================================

const eventLog =
    document.getElementById("eventLog");

const eventCount =
    document.getElementById("eventCount");

let totalEvents = 0;

let faceWasDetected = false;

let previousEyesClosed = false;


// ========================================
// VARIABLES
// ========================================

let cameraStream = null;

let timerInterval = null;

let monitoringStartTime = null;

let faceLandmarker = null;

let detectionRunning = false;

let lastVideoTime = -1;


// ========================================
// DROWSINESS VARIABLES
// ========================================

let eyesClosed = false;

let eyesClosedStartTime = null;

let totalEyesClosedMilliseconds = 0;

let lastClosedUpdateTime = null;

let drowsinessEventCount = 0;


// ========================================
// ALARM VARIABLES
// ========================================

let audioContext = null;

let alarmInterval = null;

let alarmActive = false;

let alarmActivationCount = 0;


// ========================================
// EAR GRAPH
// ========================================

let earHistory = [];

const MAX_GRAPH_POINTS = 80;

const GRAPH_MIN_EAR = 0.05;

const GRAPH_MAX_EAR = 0.45;


// ========================================
// THRESHOLDS
// ========================================

const MEDIUM_DROWSINESS_TIME =
    1000;

const HIGH_DROWSINESS_TIME =
    2000;

const EYE_CLOSED_THRESHOLD =
    0.20;


// ========================================
// EYE LANDMARKS
// ========================================

const RIGHT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
];

const LEFT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
];


// ========================================
// EVENT LOG FUNCTION
// ========================================

function addEvent(
    message,
    type = "info",
    icon = "•"
) {

    const now =
        new Date();


    const time =
        now.toLocaleTimeString(
            [],
            {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit"
            }
        );


    const emptyMessage =
        eventLog.querySelector(
            ".empty-log"
        );


    if (emptyMessage) {

        emptyMessage.remove();

    }


    const eventItem =
        document.createElement("div");


    eventItem.className =
        `event-item event-${type}`;


    eventItem.innerHTML = `

        <span class="event-time">
            ${time}
        </span>

        <span class="event-icon">
            ${icon}
        </span>

        <span class="event-text">
            ${message}
        </span>

    `;


    eventLog.prepend(
        eventItem
    );


    totalEvents++;


    eventCount.textContent =
        `${totalEvents} ${
            totalEvents === 1
                ? "Event"
                : "Events"
        }`;
}


// ========================================
// CLEAR EVENT LOG
// ========================================

function clearEventLog() {

    eventLog.innerHTML = `

        <div class="empty-log">
            No events recorded yet.
        </div>

    `;


    totalEvents = 0;


    eventCount.textContent =
        "0 Events";


    faceWasDetected =
        false;


    previousEyesClosed =
        false;
}


// ========================================
// DISTANCE
// ========================================

function distance(point1, point2) {

    const x =
        point1.x - point2.x;

    const y =
        point1.y - point2.y;

    const z =
        point1.z - point2.z;


    return Math.sqrt(
        x * x +
        y * y +
        z * z
    );
}


// ========================================
// EAR
// ========================================

function calculateEAR(
    landmarks,
    eye
) {

    const p1 =
        landmarks[eye[0]];

    const p2 =
        landmarks[eye[1]];

    const p3 =
        landmarks[eye[2]];

    const p4 =
        landmarks[eye[3]];

    const p5 =
        landmarks[eye[4]];

    const p6 =
        landmarks[eye[5]];


    const vertical1 =
        distance(p2, p6);

    const vertical2 =
        distance(p3, p5);

    const horizontal =
        distance(p1, p4);


    if (horizontal === 0) {

        return 0;
    }


    return (
        vertical1 +
        vertical2
    ) / (
        2 * horizontal
    );
}


// ========================================
// FORMAT TIME
// ========================================

function formatTime(milliseconds) {

    const totalSeconds =
        Math.floor(
            milliseconds / 1000
        );


    const minutes =
        Math.floor(
            totalSeconds / 60
        );


    const seconds =
        totalSeconds % 60;


    return (
        String(minutes)
            .padStart(2, "0") +
        ":" +
        String(seconds)
            .padStart(2, "0")
    );
}


// ========================================
// GRAPH
// ========================================

function resetGraph() {

    earHistory = [];

    drawGraph();
}


function addEARToGraph(value) {

    earHistory.push(value);


    if (
        earHistory.length >
        MAX_GRAPH_POINTS
    ) {

        earHistory.shift();

    }


    drawGraph();
}


function drawGraph() {

    const width =
        earGraph.width;

    const height =
        earGraph.height;


    graphContext.clearRect(
        0,
        0,
        width,
        height
    );


    graphContext.fillStyle =
        "#020617";

    graphContext.fillRect(
        0,
        0,
        width,
        height
    );


    graphContext.strokeStyle =
        "#1e293b";

    graphContext.lineWidth = 1;


    for (
        let i = 1;
        i < 5;
        i++
    ) {

        const y =
            (height / 5) * i;


        graphContext.beginPath();

        graphContext.moveTo(
            0,
            y
        );

        graphContext.lineTo(
            width,
            y
        );

        graphContext.stroke();
    }


    const thresholdY =
        height -
        (
            (
                EYE_CLOSED_THRESHOLD -
                GRAPH_MIN_EAR
            ) /
            (
                GRAPH_MAX_EAR -
                GRAPH_MIN_EAR
            )
        ) *
        height;


    graphContext.strokeStyle =
        "#ef4444";

    graphContext.setLineDash([
        6,
        6
    ]);


    graphContext.beginPath();

    graphContext.moveTo(
        0,
        thresholdY
    );

    graphContext.lineTo(
        width,
        thresholdY
    );

    graphContext.stroke();

    graphContext.setLineDash([]);


    if (
        earHistory.length === 0
    ) {

        graphContext.fillStyle =
            "#64748b";

        graphContext.font =
            "14px Arial";

        graphContext.textAlign =
            "center";


        graphContext.fillText(
            "Waiting for eye detection...",
            width / 2,
            height / 2
        );


        return;
    }


    graphContext.beginPath();


    earHistory.forEach(
        function(value, index) {

            const x =
                index *
                (
                    width /
                    Math.max(
                        MAX_GRAPH_POINTS - 1,
                        1
                    )
                );


            let normalized =
                (
                    value -
                    GRAPH_MIN_EAR
                ) /
                (
                    GRAPH_MAX_EAR -
                    GRAPH_MIN_EAR
                );


            normalized =
                Math.max(
                    0,
                    Math.min(
                        1,
                        normalized
                    )
                );


            const y =
                height -
                normalized *
                height;


            if (index === 0) {

                graphContext.moveTo(
                    x,
                    y
                );

            }

            else {

                graphContext.lineTo(
                    x,
                    y
                );

            }

        }
    );


    graphContext.strokeStyle =
        "#38bdf8";

    graphContext.lineWidth = 3;

    graphContext.stroke();
}


// ========================================
// STATISTICS
// ========================================

function updateStatistics() {

    drowsinessEvents.textContent =
        drowsinessEventCount;


    alarmActivations.textContent =
        alarmActivationCount;


    eyesClosedTime.textContent =
        formatTime(
            totalEyesClosedMilliseconds
        );


    currentStatus.textContent =
        driverStatus.textContent;
}


// ========================================
// ALARM SOUND
// ========================================

function playAlarmSound() {

    if (!audioContext) {

        return;
    }


    const oscillator =
        audioContext.createOscillator();

    const gainNode =
        audioContext.createGain();


    oscillator.type =
        "square";

    oscillator.frequency.value =
        880;

    gainNode.gain.value =
        0.15;


    oscillator.connect(
        gainNode
    );

    gainNode.connect(
        audioContext.destination
    );


    oscillator.start();


    oscillator.stop(
        audioContext.currentTime +
        0.35
    );
}


// ========================================
// START ALARM
// ========================================

function startAlarm() {

    if (alarmActive) {

        return;
    }


    alarmActive = true;

    alarmActivationCount++;


    alarmStatus.textContent =
        "ON";

    alarmStatus.style.color =
        "#ef4444";


    addEvent(
        "Alarm Activated",
        "danger",
        "🔊"
    );


    playAlarmSound();


    alarmInterval =
        setInterval(
            function() {

                if (alarmActive) {

                    playAlarmSound();

                }

            },
            700
        );


    updateStatistics();
}


// ========================================
// STOP ALARM
// ========================================

function stopAlarm() {

    alarmActive = false;


    if (
        alarmInterval !== null
    ) {

        clearInterval(
            alarmInterval
        );

        alarmInterval = null;
    }


    alarmStatus.textContent =
        "OFF";

    alarmStatus.style.color =
        "#ffffff";
}


// ========================================
// NORMAL
// ========================================

function setNormalStatus() {

    const wasDrowsy =
        driverStatus.textContent ===
        "DROWSY";


    driverStatus.textContent =
        "NORMAL";


    statusMessage.textContent =
        "Driver appears to be awake";


    drowsinessLevel.textContent =
        "LOW";


    alarmStatus.textContent =
        "OFF";


    statusIcon.textContent =
        "✓";


    driverStatusBox.style.background =
        "rgba(34, 197, 94, 0.10)";


    driverStatusBox.style.border =
        "1px solid rgba(34, 197, 94, 0.35)";


    driverStatusBox.style.boxShadow =
        "none";


    statusIcon.style.background =
        "#22c55e";

    driverStatus.style.color =
        "#22c55e";

    drowsinessLevel.style.color =
        "#ffffff";


    if (wasDrowsy) {

        addEvent(
            "Driver Alert - Eyes Open",
            "normal",
            "✓"
        );

    }


    stopAlarm();

    updateStatistics();
}


// ========================================
// MEDIUM
// ========================================

function setMediumDrowsiness() {

    const wasNormal =
        driverStatus.textContent ===
        "NORMAL";


    driverStatus.textContent =
        "WARNING";


    statusMessage.textContent =
        "Eyes closed for a short period";


    drowsinessLevel.textContent =
        "MEDIUM";


    alarmStatus.textContent =
        "OFF";


    statusIcon.textContent =
        "!";


    driverStatusBox.style.background =
        "rgba(245, 158, 11, 0.12)";


    driverStatusBox.style.border =
        "1px solid rgba(245, 158, 11, 0.5)";


    driverStatusBox.style.boxShadow =
        "0 0 20px rgba(245, 158, 11, 0.15)";


    statusIcon.style.background =
        "#f59e0b";

    driverStatus.style.color =
        "#f59e0b";

    drowsinessLevel.style.color =
        "#f59e0b";


    if (wasNormal) {

        addEvent(
            "Drowsiness Warning",
            "warning",
            "⚠️"
        );

    }


    stopAlarm();

    updateStatistics();
}


// ========================================
// HIGH
// ========================================

function setHighDrowsiness() {

    driverStatus.textContent =
        "DROWSY";


    statusMessage.textContent =
        "⚠ DROWSINESS DETECTED";


    drowsinessLevel.textContent =
        "HIGH";


    alarmStatus.textContent =
        "ON";


    statusIcon.textContent =
        "!";


    driverStatusBox.style.background =
        "rgba(239, 68, 68, 0.15)";


    driverStatusBox.style.border =
        "2px solid #ef4444";


    driverStatusBox.style.boxShadow =
        "0 0 30px rgba(239, 68, 68, 0.25)";


    statusIcon.style.background =
        "#ef4444";

    driverStatus.style.color =
        "#ef4444";

    drowsinessLevel.style.color =
        "#ef4444";

    alarmStatus.style.color =
        "#ef4444";


    if (!alarmActive) {

        drowsinessEventCount++;


        addEvent(
            "Drowsiness Detected",
            "danger",
            "🚨"
        );

    }


    startAlarm();

    updateStatistics();
}


// ========================================
// DROWSINESS UPDATE
// ========================================

function updateDrowsiness() {

    if (!eyesClosed) {

        eyesClosedStartTime =
            null;

        lastClosedUpdateTime =
            null;


        setNormalStatus();

        return;
    }


    if (
        eyesClosedStartTime ===
        null
    ) {

        eyesClosedStartTime =
            Date.now();

        lastClosedUpdateTime =
            Date.now();
    }


    const now =
        Date.now();


    if (
        lastClosedUpdateTime !==
        null
    ) {

        totalEyesClosedMilliseconds +=
            now -
            lastClosedUpdateTime;
    }


    lastClosedUpdateTime =
        now;


    const closedDuration =
        now -
        eyesClosedStartTime;


    if (
        closedDuration >=
        HIGH_DROWSINESS_TIME
    ) {

        setHighDrowsiness();

    }

    else if (
        closedDuration >=
        MEDIUM_DROWSINESS_TIME
    ) {

        setMediumDrowsiness();

    }

    else {

        driverStatus.textContent =
            "NORMAL";

        statusMessage.textContent =
            "Monitoring eye closure";

        drowsinessLevel.textContent =
            "LOW";


        stopAlarm();

        updateStatistics();
    }
}


// ========================================
// EAR DISPLAY
// ========================================

function updateEARDisplay(
    averageEAR,
    isClosed
) {

    earValue.textContent =
        averageEAR.toFixed(3);


    addEARToGraph(
        averageEAR
    );


    if (isClosed) {

        earState.textContent =
            "Eyes Closed";

        earState.style.color =
            "#ef4444";

        earValue.style.color =
            "#ef4444";

    }

    else {

        earState.textContent =
            "Eyes Open";

        earState.style.color =
            "#22c55e";

        earValue.style.color =
            "#38bdf8";
    }
}


// ========================================
// LOAD FACE MODEL
// ========================================

async function loadFaceModel() {

    faceStatus.textContent =
        "Loading AI...";

    faceStatus.style.color =
        "#f59e0b";


    try {

        const vision =
            await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
            );


        faceLandmarker =
            await FaceLandmarker.createFromOptions(
                vision,
                {

                    baseOptions: {

                        modelAssetPath:
                            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

                    },


                    runningMode:
                        "VIDEO",

                    numFaces:
                        1,

                    minFaceDetectionConfidence:
                        0.5,

                    minFacePresenceConfidence:
                        0.5,

                    minTrackingConfidence:
                        0.5

                }
            );


        faceStatus.textContent =
            "AI Ready";

        faceStatus.style.color =
            "#22c55e";


        return true;

    }

    catch (error) {

        console.error(
            "FACE LANDMARKER ERROR:",
            error
        );


        faceStatus.textContent =
            "AI Error";

        faceStatus.style.color =
            "#ef4444";


        return false;
    }
}


// ========================================
// DETECTION LOOP
// ========================================

function detectFaceAndEyes() {

    if (!detectionRunning) {

        return;
    }


    if (
        faceLandmarker !== null &&
        camera.readyState >= 2
    ) {

        try {

            if (
                camera.currentTime !==
                lastVideoTime
            ) {

                lastVideoTime =
                    camera.currentTime;


                const results =
                    faceLandmarker.detectForVideo(
                        camera,
                        performance.now()
                    );


                if (
                    results.faceLandmarks &&
                    results.faceLandmarks.length > 0
                ) {

                    const landmarks =
                        results.faceLandmarks[0];


                    faceStatus.textContent =
                        "Face Detected";


                    faceStatus.style.color =
                        "#22c55e";


                    if (!faceWasDetected) {

                        faceWasDetected =
                            true;


                        addEvent(
                            "Face Detected",
                            "info",
                            "👤"
                        );

                    }


                    const rightEAR =
                        calculateEAR(
                            landmarks,
                            RIGHT_EYE
                        );


                    const leftEAR =
                        calculateEAR(
                            landmarks,
                            LEFT_EYE
                        );


                    const averageEAR =
                        (
                            rightEAR +
                            leftEAR
                        ) / 2;


                    if (
                        averageEAR >
                        EYE_CLOSED_THRESHOLD
                    ) {

                        eyesClosed =
                            false;


                        eyeStatus.textContent =
                            "OPEN";


                        eyeStatus.style.color =
                            "#22c55e";


                        eyeDetection.textContent =
                            "Eyes Open";


                        eyeDetection.style.color =
                            "#22c55e";


                        if (
                            previousEyesClosed
                        ) {

                            addEvent(
                                "Eyes Open",
                                "normal",
                                "👁️"
                            );

                        }


                        previousEyesClosed =
                            false;


                        updateEARDisplay(
                            averageEAR,
                            false
                        );


                        updateDrowsiness();

                    }

                    else {

                        if (!eyesClosed) {

                            eyesClosed =
                                true;


                            eyesClosedStartTime =
                                Date.now();


                            lastClosedUpdateTime =
                                Date.now();

                        }


                        eyeStatus.textContent =
                            "CLOSED";


                        eyeStatus.style.color =
                            "#ef4444";


                        eyeDetection.textContent =
                            "Eyes Closed";


                        eyeDetection.style.color =
                            "#ef4444";


                        if (
                            !previousEyesClosed
                        ) {

                            addEvent(
                                "Eyes Closed",
                                "warning",
                                "👁️"
                            );

                        }


                        previousEyesClosed =
                            true;


                        updateEARDisplay(
                            averageEAR,
                            true
                        );


                        updateDrowsiness();

                    }

                }

                else {

                    faceStatus.textContent =
                        "No Face";


                    faceStatus.style.color =
                        "#ef4444";


                    if (faceWasDetected) {

                        addEvent(
                            "Face Lost",
                            "warning",
                            "👤"
                        );

                    }


                    faceWasDetected =
                        false;


                    eyeDetection.textContent =
                        "Face Required";


                    eyeDetection.style.color =
                        "#f59e0b";


                    eyeStatus.textContent =
                        "NO FACE";


                    eyeStatus.style.color =
                        "#ef4444";


                    earValue.textContent =
                        "0.00";


                    earState.textContent =
                        "Face not detected";


                    earState.style.color =
                        "#f59e0b";


                    eyesClosed =
                        false;

                    eyesClosedStartTime =
                        null;

                    lastClosedUpdateTime =
                        null;


                    setNormalStatus();
                }

            }

        }

        catch (error) {

            console.error(
                "DETECTION ERROR:",
                error
            );
        }
    }


    updateStatistics();


    requestAnimationFrame(
        detectFaceAndEyes
    );
}


// ========================================
// START MONITORING
// ========================================

startBtn.addEventListener(
    "click",
    async function() {

        if (
            cameraStream !== null
        ) {

            return;
        }


        try {

            if (!audioContext) {

                audioContext =
                    new (
                        window.AudioContext ||
                        window.webkitAudioContext
                    )();
            }


            if (
                audioContext.state ===
                "suspended"
            ) {

                await audioContext.resume();
            }


            cameraStream =
                await navigator.mediaDevices
                    .getUserMedia({

                        video: true,

                        audio: false

                    });


            camera.srcObject =
                cameraStream;


            camera.style.display =
                "block";


            cameraPlaceholder.style.display =
                "none";


            cameraStatus.textContent =
                "Camera Connected";

            cameraStatus.style.color =
                "#22c55e";


            systemStatus.textContent =
                "ACTIVE";

            headerStatus.textContent =
                "System Active";


            monitoringStartTime =
                Date.now();


            timer.textContent =
                "00:00";


            timerInterval =
                setInterval(
                    function() {

                        const elapsed =
                            Date.now() -
                            monitoringStartTime;


                        const totalSeconds =
                            Math.floor(
                                elapsed / 1000
                            );


                        const minutes =
                            Math.floor(
                                totalSeconds / 60
                            );


                        const seconds =
                            totalSeconds % 60;


                        timer.textContent =
                            String(minutes)
                                .padStart(2, "0") +
                            ":" +
                            String(seconds)
                                .padStart(2, "0");

                    },
                    250
                );


            // Reset statistics

            drowsinessEventCount = 0;

            alarmActivationCount = 0;

            totalEyesClosedMilliseconds = 0;


            drowsinessEvents.textContent =
                "0";

            alarmActivations.textContent =
                "0";

            eyesClosedTime.textContent =
                "00:00";

            currentStatus.textContent =
                "NORMAL";


            // Reset AI

            earValue.textContent =
                "0.00";

            earState.textContent =
                "Waiting for detection";

            earState.style.color =
                "#64748b";

            earValue.style.color =
                "#38bdf8";


            resetGraph();


            // Reset event log

            clearEventLog();


            addEvent(
                "Monitoring Started",
                "info",
                "▶"
            );


            // Reset drowsiness

            eyesClosed = false;

            eyesClosedStartTime = null;

            lastClosedUpdateTime = null;

            previousEyesClosed = false;

            faceWasDetected = false;


            stopAlarm();

            setNormalStatus();


            const loaded =
                await loadFaceModel();


            if (!loaded) {

                return;
            }


            detectionRunning =
                true;


            lastVideoTime =
                -1;


            detectFaceAndEyes();

        }

        catch (error) {

            console.error(
                "CAMERA ERROR:",
                error
            );


            cameraStream =
                null;


            stopAlarm();


            alert(
                "Unable to access the camera. Please allow camera permission."
            );
        }

    }
);


// ========================================
// STOP MONITORING
// ========================================

stopBtn.addEventListener(
    "click",
    function() {

        if (
            cameraStream !== null
        ) {

            addEvent(
                "Monitoring Stopped",
                "info",
                "■"
            );
        }


        detectionRunning =
            false;


        stopAlarm();


        if (
            timerInterval !== null
        ) {

            clearInterval(
                timerInterval
            );

            timerInterval =
                null;
        }


        monitoringStartTime =
            null;


        if (
            cameraStream !== null
        ) {

            cameraStream
                .getTracks()
                .forEach(
                    function(track) {

                        track.stop();

                    }
                );


            cameraStream =
                null;
        }


        camera.srcObject =
            null;


        camera.style.display =
            "none";


        cameraPlaceholder.style.display =
            "flex";


        timer.textContent =
            "00:00";


        systemStatus.textContent =
            "READY";


        headerStatus.textContent =
            "System Ready";


        cameraStatus.textContent =
            "Camera Offline";


        cameraStatus.style.color =
            "#f59e0b";


        faceStatus.textContent =
            "Waiting";


        faceStatus.style.color =
            "";


        eyeDetection.textContent =
            "Waiting";


        eyeDetection.style.color =
            "";


        eyeStatus.textContent =
            "WAITING";


        eyeStatus.style.color =
            "";


        earValue.textContent =
            "0.00";


        earState.textContent =
            "Waiting for detection";


        earState.style.color =
            "#64748b";


        earValue.style.color =
            "#38bdf8";


        resetGraph();


        eyesClosed = false;

        eyesClosedStartTime = null;

        lastClosedUpdateTime = null;

        previousEyesClosed = false;

        faceWasDetected = false;


        setNormalStatus();


        // Reset statistics

        drowsinessEventCount = 0;

        alarmActivationCount = 0;

        totalEyesClosedMilliseconds = 0;


        drowsinessEvents.textContent =
            "0";

        alarmActivations.textContent =
            "0";

        eyesClosedTime.textContent =
            "00:00";

        currentStatus.textContent =
            "NORMAL";

    }
);


// ========================================
// INITIAL GRAPH
// ========================================

resetGraph();