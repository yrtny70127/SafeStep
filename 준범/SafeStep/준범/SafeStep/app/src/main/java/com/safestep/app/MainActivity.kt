package com.safestep.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.provider.Settings
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.safestep.app.detect.Detection
import com.safestep.app.detect.ObjectDetector
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    // ── Views ─────────────────────────────────────────────────────────────────
    private lateinit var rootLayout: ConstraintLayout
    private lateinit var previewView: PreviewView
    private lateinit var warningBanner: LinearLayout
    private lateinit var warningText: TextView
    private lateinit var statusText: TextView
    private lateinit var micButton: Button
    private lateinit var permissionDeniedView: LinearLayout
    private lateinit var goToSettingsButton: Button

    // ── Core components ────────────────────────────────────────────────────────
    private lateinit var tts: TextToSpeech
    private lateinit var cameraExecutor: ExecutorService
    private var vibrator: Vibrator? = null
    private lateinit var detector: ObjectDetector

    // ── Gesture ────────────────────────────────────────────────────────────────
    private lateinit var gestureDetector: GestureDetector

    // ── Speech recognition ─────────────────────────────────────────────────────
    private var speechRecognizer: SpeechRecognizer? = null
    private var isListening = false

    // ── State ──────────────────────────────────────────────────────────────────
    /** 마지막으로 음성 안내한 메시지와 시각 */
    private var lastSpeakTimeMs: Long = 0L
    private var lastSpoken: String = ""

    /** 현재 프레임에서 감지된 모든 객체 (음성 명령 응답에 사용) */
    @Volatile private var currentDetections: List<Detection> = emptyList()

    /** 마지막 경고 메시지 (더블탭 재발화에 사용) */
    @Volatile private var lastWarningMessage: String = ""

    companion object {
        private const val TAG = "SafeStepMain"
        private const val REQUEST_CAMERA_MIC = 100

        /** 박스가 화면의 이 비율 이상이면 위험 감지 시작 */
        private const val DANGER_AREA = 0.20f
        /** 이 비율 이상이면 "매우 가까이" */
        private const val VERY_CLOSE_AREA = 0.40f
        /** 같은 메시지 재발화 최소 간격 (ms) */
        private const val SPEAK_COOLDOWN_MS = 2500L
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Lifecycle
    // ══════════════════════════════════════════════════════════════════════════

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        rootLayout         = findViewById(R.id.rootLayout)
        previewView        = findViewById(R.id.previewView)
        warningBanner      = findViewById(R.id.warningBanner)
        warningText        = findViewById(R.id.warningText)
        statusText         = findViewById(R.id.statusText)
        micButton          = findViewById(R.id.micButton)
        permissionDeniedView  = findViewById(R.id.permissionDeniedView)
        goToSettingsButton = findViewById(R.id.goToSettingsButton)

        tts = TextToSpeech(this, this)
        @Suppress("DEPRECATION")
        vibrator = getSystemService(VIBRATOR_SERVICE) as Vibrator
        cameraExecutor = Executors.newSingleThreadExecutor()
        detector = ObjectDetector.create(this)

        setupGestureDetector()
        setupMicButton()
        setupSettingsButton()

        val permissions = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        if (hasAllPermissions(permissions)) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, permissions, REQUEST_CAMERA_MIC)
        }
    }

    override fun onResume() {
        super.onResume()
        // 설정에서 돌아왔을 때 권한이 허용됐으면 자동 시작
        if (hasCameraPermission() && permissionDeniedView.visibility == View.VISIBLE) {
            permissionDeniedView.visibility = View.GONE
            startCamera()
        }
    }

    override fun onDestroy() {
        cameraExecutor.shutdown()
        tts.shutdown()
        speechRecognizer?.destroy()
        runCatching { detector.close() }
        super.onDestroy()
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Permissions
    // ══════════════════════════════════════════════════════════════════════════

    private fun hasAllPermissions(permissions: Array<String>) =
        permissions.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != REQUEST_CAMERA_MIC) return

        if (hasCameraPermission()) {
            permissionDeniedView.visibility = View.GONE
            startCamera()
        } else {
            showPermissionDenied()
        }
    }

    private fun showPermissionDenied() {
        permissionDeniedView.visibility = View.VISIBLE
        tts.speak(
            "카메라 권한이 필요합니다. 설정으로 이동 버튼을 눌러 카메라를 허용해주세요.",
            TextToSpeech.QUEUE_FLUSH, null, "perm-denied"
        )
    }

    private fun setupSettingsButton() {
        goToSettingsButton.setOnClickListener {
            val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                data = Uri.fromParts("package", packageName, null)
            }
            startActivity(intent)
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Gesture — 더블탭으로 현재 상태 재발화
    // ══════════════════════════════════════════════════════════════════════════

    private fun setupGestureDetector() {
        gestureDetector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDoubleTap(e: MotionEvent): Boolean {
                repeatLastStatus()
                return true
            }
        })
        rootLayout.setOnTouchListener { _, event ->
            gestureDetector.onTouchEvent(event)
            true
        }
    }

    private fun repeatLastStatus() {
        val msg = if (lastWarningMessage.isNotEmpty()) {
            lastWarningMessage
        } else {
            "현재 위험 없음"
        }
        tts.speak(msg, TextToSpeech.QUEUE_FLUSH, null, "repeat-${SystemClock.elapsedRealtime()}")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Camera
    // ══════════════════════════════════════════════════════════════════════════

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(cameraExecutor, ::analyzeFrame) }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                runOnUiThread { statusText.text = "감지 중" }
            } catch (e: Exception) {
                Log.e(TAG, "카메라 바인딩 실패", e)
                runOnUiThread { statusText.text = "카메라 오류" }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @androidx.annotation.OptIn(androidx.camera.core.ExperimentalGetImage::class)
    private fun analyzeFrame(imageProxy: ImageProxy) {
        try {
            val bitmap = imageProxy.toBitmap()
            val detections = detector.detect(bitmap, imageProxy.imageInfo.rotationDegrees)
            currentDetections = detections
            handleDetections(detections)
        } catch (e: Exception) {
            Log.e(TAG, "프레임 분석 실패", e)
        } finally {
            imageProxy.close()
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Detection → Warning
    // ══════════════════════════════════════════════════════════════════════════

    private fun handleDetections(detections: List<Detection>) {
        if (detections.isEmpty()) return

        val worst = detections.maxByOrNull { d ->
            val centerWeight = 1f - kotlin.math.abs(d.centerX() - 0.5f)
            d.area() * (0.7f + 0.3f * centerWeight) * d.confidence
        } ?: return

        if (worst.area() < DANGER_AREA) return

        val koLabel = labelToKorean(worst.label)
        val side = when {
            worst.centerX() < 0.33f -> "왼쪽"
            worst.centerX() > 0.66f -> "오른쪽"
            else -> "정면"
        }
        val distance = when {
            worst.area() >= VERY_CLOSE_AREA -> "매우 가까이 "
            else -> ""
        }
        val message = "$side $distance$koLabel 접근"
        lastWarningMessage = message
        showWarning(message, side)
    }

    // 서버가 한국어 label_ko 를 Detection.label 에 담아서 보내주므로 변환 불필요
    private fun labelToKorean(label: String): String = label

    @Suppress("unused")
    private fun labelToKoreanLegacy(label: String): String = when (label.lowercase()) {
        "person"                    -> "사람"
        "car", "truck", "bus"       -> "차량"
        "bicycle", "motorcycle"     -> "이륜차"
        "chair"                     -> "의자"
        "pole", "tree"              -> "기둥"
        "dog", "cat"                -> "동물"
        "fire hydrant"              -> "소화전"
        "traffic light"             -> "신호등"
        else                        -> label
    }

    fun showWarning(message: String, side: String = "정면") {
        runOnUiThread {
            warningText.text = message
            warningBanner.visibility = View.VISIBLE
        }
        speak(message)
        vibrateForDirection(side)
        warningBanner.postDelayed({
            runOnUiThread { warningBanner.visibility = View.GONE }
        }, 3000)
    }

    // ══════════════════════════════════════════════════════════════════════════
    // TTS
    // ══════════════════════════════════════════════════════════════════════════

    private fun speak(text: String) {
        val now = SystemClock.elapsedRealtime()
        if (text == lastSpoken && now - lastSpeakTimeMs < SPEAK_COOLDOWN_MS) return
        lastSpoken = text
        lastSpeakTimeMs = now
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "safestep-$now")
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) tts.language = Locale.KOREAN
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Vibration — 방향별 패턴 차별화
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * 왼쪽 : 짧-짧-긴  (왼쪽으로 주의 유도)
     * 정면 : 긴-짧-긴  (강한 경고)
     * 오른쪽: 긴-짧-짧 (오른쪽으로 주의 유도)
     */
    private fun vibrateForDirection(side: String) {
        val pattern: LongArray = when (side) {
            "왼쪽"  -> longArrayOf(0, 120, 80, 120, 80, 350)
            "오른쪽" -> longArrayOf(0, 350, 80, 120, 80, 120)
            else    -> longArrayOf(0, 250, 80, 100, 80, 250) // 정면: 대칭 강 패턴
        }
        @Suppress("DEPRECATION")
        vibrator?.vibrate(VibrationEffect.createWaveform(pattern, -1))
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Voice command — "주변 알려줘"
    // ══════════════════════════════════════════════════════════════════════════

    private fun setupMicButton() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            micButton.visibility = View.GONE
            return
        }
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this).apply {
            setRecognitionListener(recognitionListener)
        }
        micButton.setOnClickListener {
            if (isListening) stopListening() else startListening()
        }
    }

    private fun startListening() {
        isListening = true
        micButton.text = "⏹"
        micButton.contentDescription = "음성 인식 중. 주변 알려줘 라고 말하세요."
        tts.speak("듣고 있어요.", TextToSpeech.QUEUE_FLUSH, null, "mic-start")

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
        }
        speechRecognizer?.startListening(intent)
    }

    private fun stopListening() {
        isListening = false
        micButton.text = "🎙"
        micButton.contentDescription = "음성 명령 버튼. 누른 후 주변 알려줘 라고 말하세요."
        speechRecognizer?.stopListening()
    }

    private fun handleVoiceCommand(text: String) {
        Log.d(TAG, "음성 인식 결과: $text")
        when {
            text.contains("주변") || text.contains("알려") || text.contains("뭐가") -> {
                speakCurrentDetections()
            }
            text.contains("멈춰") || text.contains("중지") || text.contains("꺼") -> {
                tts.speak("안내를 잠시 멈춥니다.", TextToSpeech.QUEUE_FLUSH, null, "stop-cmd")
            }
            else -> {
                tts.speak("알 수 없는 명령입니다. 주변 알려줘 라고 말해보세요.", TextToSpeech.QUEUE_FLUSH, null, "unknown-cmd")
            }
        }
    }

    private fun speakCurrentDetections() {
        val detections = currentDetections
        if (detections.isEmpty()) {
            tts.speak("현재 주변에 감지된 물체가 없습니다.", TextToSpeech.QUEUE_FLUSH, null, "env-empty")
            return
        }
        val items = detections
            .sortedByDescending { it.area() }
            .take(5)
            .map { d ->
                val side = when {
                    d.centerX() < 0.33f -> "왼쪽"
                    d.centerX() > 0.66f -> "오른쪽"
                    else -> "정면"
                }
                "$side ${labelToKorean(d.label)}"
            }
            .distinct()
            .joinToString(", ")
        tts.speak("감지된 물체: $items", TextToSpeech.QUEUE_FLUSH, null, "env-report")
    }

    private val recognitionListener = object : RecognitionListener {
        override fun onResults(results: Bundle?) {
            isListening = false
            micButton.text = "🎙"
            val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
            val best = matches?.firstOrNull() ?: return
            handleVoiceCommand(best)
        }
        override fun onError(error: Int) {
            isListening = false
            micButton.text = "🎙"
            val msg = when (error) {
                SpeechRecognizer.ERROR_NO_MATCH       -> "인식된 음성이 없습니다."
                SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "음성 입력 시간이 초과됐습니다."
                SpeechRecognizer.ERROR_NETWORK        -> "네트워크 오류가 발생했습니다."
                else                                  -> "음성 인식 오류가 발생했습니다."
            }
            tts.speak(msg, TextToSpeech.QUEUE_FLUSH, null, "mic-err")
        }
        override fun onReadyForSpeech(params: Bundle?) {}
        override fun onBeginningOfSpeech() {}
        override fun onRmsChanged(rmsdB: Float) {}
        override fun onBufferReceived(buffer: ByteArray?) {}
        override fun onEndOfSpeech() {}
        override fun onPartialResults(partialResults: Bundle?) {}
        override fun onEvent(eventType: Int, params: Bundle?) {}
    }
}
