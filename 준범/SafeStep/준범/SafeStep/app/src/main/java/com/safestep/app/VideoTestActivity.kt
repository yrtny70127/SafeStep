package com.safestep.app

import android.content.Intent
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.tts.TextToSpeech
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.VideoView
import com.safestep.app.detect.RemoteDetector
import com.safestep.app.detect.SegmentationClient
import com.safestep.app.detect.SegmentResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.safestep.app.detect.Detection
import com.safestep.app.detect.ObjectDetector
import java.util.Locale

class VideoTestActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    // ── Views ─────────────────────────────────────────────────────────────────
    private lateinit var videoView: VideoView
    private lateinit var bboxOverlay: BoundingBoxOverlay
    private lateinit var warningBanner: LinearLayout
    private lateinit var warningText: TextView
    private lateinit var emptyView: LinearLayout
    private lateinit var detectionStatus: TextView
    private lateinit var selectVideoButton: Button
    private lateinit var playPauseButton: Button

    // ── Core ──────────────────────────────────────────────────────────────────
    private lateinit var tts: TextToSpeech
    private lateinit var detector: ObjectDetector
    private lateinit var segClient: SegmentationClient
    private lateinit var segmentOverlay: ImageView
    private var vibrator: Vibrator? = null
    @Volatile private var frameCount = 0
    @Volatile private var lastSurfaceStatus = ""
    @Volatile private var lastSurfaceSpeakMs = 0L

    // ── 프레임 추출 타이머 ─────────────────────────────────────────────────────
    private val frameHandler = Handler(Looper.getMainLooper())
    private var retriever: MediaMetadataRetriever? = null
    private var videoUri: Uri? = null
    private var isDetecting = false

    // ── 탐지 상태 ──────────────────────────────────────────────────────────────
    @Volatile private var lastSpoken = ""
    @Volatile private var lastSpeakMs = 0L

    companion object {
        private const val FRAME_INTERVAL_MS = 500L  // 탐지 간격 (ms)
        private const val DANGER_AREA       = 0.20f
        private const val VERY_CLOSE_AREA   = 0.40f
        private const val SPEAK_COOLDOWN_MS = 2500L
    }

    // ── 동영상 선택 런처 ──────────────────────────────────────────────────────
    private val pickVideo = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { loadVideo(it) }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Lifecycle
    // ══════════════════════════════════════════════════════════════════════════

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_test)

        videoView        = findViewById(R.id.videoView)
        bboxOverlay      = findViewById(R.id.bboxOverlay)
        warningBanner    = findViewById(R.id.warningBanner)
        warningText      = findViewById(R.id.warningText)
        emptyView        = findViewById(R.id.emptyView)
        detectionStatus  = findViewById(R.id.detectionStatus)
        selectVideoButton = findViewById(R.id.selectVideoButton)
        playPauseButton  = findViewById(R.id.playPauseButton)

        tts          = TextToSpeech(this, this)
        detector     = ObjectDetector.create(this)
        segClient    = SegmentationClient(RemoteDetector.SERVER_URL)
        segmentOverlay = findViewById(R.id.segmentOverlay)
        @Suppress("DEPRECATION")
        vibrator = getSystemService(VIBRATOR_SERVICE) as Vibrator

        setupButtons()
        setupVideoView()
    }

    override fun onPause() {
        super.onPause()
        pauseVideo()
    }

    override fun onDestroy() {
        stopFrameDetection()
        retriever?.release()
        tts.shutdown()
        runCatching { detector.close() }
        super.onDestroy()
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 버튼 설정
    // ══════════════════════════════════════════════════════════════════════════

    private fun setupButtons() {
        selectVideoButton.setOnClickListener {
            pickVideo.launch("video/*")
        }

        playPauseButton.setOnClickListener {
            if (videoView.isPlaying) pauseVideo() else playVideo()
        }

        findViewById<Button>(R.id.backButton).setOnClickListener {
            startActivity(Intent(this, SplashActivity::class.java))
            finish()
        }
    }

    private fun setupVideoView() {
        videoView.setOnCompletionListener {
            stopFrameDetection()
            playPauseButton.text = "▶ 재생"
            detectionStatus.text = "동영상 재생 완료"
            bboxOverlay.clear()
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 동영상 로드 / 재생 제어
    // ══════════════════════════════════════════════════════════════════════════

    private fun loadVideo(uri: Uri) {
        videoUri = uri
        stopFrameDetection()
        bboxOverlay.clear()

        // MediaMetadataRetriever 초기화
        retriever?.release()
        retriever = MediaMetadataRetriever().apply {
            setDataSource(this@VideoTestActivity, uri)
        }

        // VideoView 설정
        videoView.setVideoURI(uri)
        videoView.setOnPreparedListener { mp ->
            mp.isLooping = false
            emptyView.visibility = View.GONE
            playPauseButton.isEnabled = true
            detectionStatus.text = "재생 버튼을 눌러 탐지를 시작하세요"
        }
        videoView.requestFocus()
    }

    private fun playVideo() {
        videoView.start()
        playPauseButton.text = "⏸ 일시정지"
        startFrameDetection()
    }

    private fun pauseVideo() {
        if (videoView.isPlaying) videoView.pause()
        playPauseButton.text = "▶ 재생"
        stopFrameDetection()
        bboxOverlay.clear()
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 프레임 추출 → 탐지 루프
    // ══════════════════════════════════════════════════════════════════════════

    private val frameRunnable = object : Runnable {
        override fun run() {
            if (videoView.isPlaying && !isDetecting) {
                extractAndDetect(videoView.currentPosition)
            }
            frameHandler.postDelayed(this, FRAME_INTERVAL_MS)
        }
    }

    private fun startFrameDetection() {
        frameHandler.removeCallbacks(frameRunnable)
        frameHandler.post(frameRunnable)
    }

    private fun stopFrameDetection() {
        frameHandler.removeCallbacks(frameRunnable)
        isDetecting = false
    }

    private fun extractAndDetect(positionMs: Int) {
        val ret = retriever ?: return
        isDetecting = true

        Thread {
            try {
                // 현재 재생 위치 프레임 추출 (마이크로초 단위)
                val bitmap = ret.getFrameAtTime(
                    positionMs * 1000L,
                    MediaMetadataRetriever.OPTION_CLOSEST_SYNC
                )

                if (bitmap != null) {
                    val detections = detector.detect(bitmap, 0)

                    // 세그멘테이션 (2프레임마다)
                    frameCount++
                    val seg = if (frameCount % 2 == 0) segClient.segment(bitmap, 0) else null

                    runOnUiThread {
                        handleDetections(detections)
                        if (seg != null) handleSegmentation(seg)
                        isDetecting = false
                    }
                } else {
                    runOnUiThread { isDetecting = false }
                }
            } catch (e: Exception) {
                runOnUiThread { isDetecting = false }
            }
        }.start()
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 탐지 결과 처리
    // ══════════════════════════════════════════════════════════════════════════

    private fun handleDetections(detections: List<Detection>) {
        // 바운딩박스 갱신
        bboxOverlay.updateDetections(detections)

        // 상태 텍스트
        detectionStatus.text = if (detections.isEmpty()) {
            "감지된 객체 없음"
        } else {
            val labels = detections.take(3).joinToString(", ") {
                "${it.label} ${(it.confidence * 100).toInt()}%"
            }
            "감지: $labels"
        }

        // 위험 경고: 사람은 매우 가까울 때만, 그 외 장애물은 DANGER_AREA 이상
        val worst = detections.maxByOrNull { d ->
            val cw = 1f - kotlin.math.abs(d.centerX() - 0.5f)
            d.area() * (0.7f + 0.3f * cw) * d.confidence
        } ?: return
        val isPerson  = worst.label in listOf("사람", "person")
        val threshold = if (isPerson) 0.60f else DANGER_AREA
        if (worst.area() < threshold) return

        val side = when {
            worst.centerX() < 0.33f -> "왼쪽"
            worst.centerX() > 0.66f -> "오른쪽"
            else                     -> "정면"
        }
        val distTag = if (worst.area() >= VERY_CLOSE_AREA) "매우 가까이 " else ""
        val msg = "$side $distTag${worst.label} 접근"
        showWarning(msg, side)
    }

    private fun handleSegmentation(seg: SegmentResult) {
        segmentOverlay.setImageBitmap(seg.maskBitmap)

        val status = seg.status
        if (status == lastSurfaceStatus) return   // 상태 변화 없으면 무시
        lastSurfaceStatus = status

        when (status) {
            "road"    -> tts.speak("차도입니다. 주의하세요.",     TextToSpeech.QUEUE_ADD, null, "road-warn")
            "caution" -> tts.speak("위험 구역입니다. 주의하세요.", TextToSpeech.QUEUE_ADD, null, "caution-warn")
            "alley"   -> tts.speak("골목길입니다. 주의하세요.",   TextToSpeech.QUEUE_ADD, null, "alley-warn")
        }
    }

    private fun showWarning(message: String, side: String) {
        warningText.text = message
        warningBanner.visibility = View.VISIBLE

        val now = SystemClock.elapsedRealtime()
        if (message != lastSpoken || now - lastSpeakMs >= SPEAK_COOLDOWN_MS) {
            lastSpoken  = message
            lastSpeakMs = now
            tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, "warn-$now")
        }

        val pattern = when (side) {
            "왼쪽"  -> longArrayOf(0, 120, 80, 120, 80, 350)
            "오른쪽" -> longArrayOf(0, 350, 80, 120, 80, 120)
            else    -> longArrayOf(0, 250, 80, 100, 80, 250)
        }
        @Suppress("DEPRECATION")
        vibrator?.vibrate(VibrationEffect.createWaveform(pattern, -1))

        warningBanner.postDelayed({ warningBanner.visibility = View.GONE }, 3_000)
    }

    // ══════════════════════════════════════════════════════════════════════════
    // TTS
    // ══════════════════════════════════════════════════════════════════════════

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.KOREAN
            tts.speak(
                "동영상 테스트 모드입니다. 동영상 선택 버튼을 눌러주세요.",
                TextToSpeech.QUEUE_FLUSH, null, "video-intro"
            )
        }
    }
}
