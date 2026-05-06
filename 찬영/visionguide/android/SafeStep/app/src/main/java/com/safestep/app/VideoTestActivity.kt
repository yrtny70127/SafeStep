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
    private var remoteDetector: RemoteDetector? = null  // 캐스팅 캐시
    private lateinit var segClient: SegmentationClient
    private var vibrator: Vibrator? = null
    @Volatile private var frameCount = 0
    @Volatile private var lastSurfaceStatus = ""

    // ── 프레임 추출 타이머 ─────────────────────────────────────────────────────
    private val frameHandler = Handler(Looper.getMainLooper())
    private var retriever: MediaMetadataRetriever? = null
    private var videoUri: Uri? = null
    private var isDetecting = false

    // ── Depth 상태 ────────────────────────────────────────────────────────────
    @Volatile private var depthNullStreak = 0
    @Volatile private var depthUnavailSpoken = false

    // ── 진동 / 차도 반복 컨트롤러 (GuideHelper.kt) ─────────────────────────
    private lateinit var vibCtrl:  VibrationController
    private lateinit var roadCtrl: RoadRepeatController

    // ── 신호등 색상 (횡단보도) ────────────────────────────────────────────────
    // "unset": 횡단보도 미진입 or 초기값 / "": 신호 인식 불가 / "red"/"green": 신호
    @Volatile private var lastTrafficLightColor = "unset"

    // ── 차량 근접 여부 (횡단보도 "건너셔도 됩니다" 판단용) ──────────────────
    @Volatile private var hasNearbyVehicle = false

    // ── 거리 기반 음성 쿨다운 ─────────────────────────────────────────────────
    @Volatile private var alert5mFired = false
    @Volatile private var alert1mFired = false
    @Volatile private var lastTrackedLabel = ""
    // 1m 긴급: 같은 라벨 3초 쿨다운 (명세 1순위)
    @Volatile private var lastUrgentLabel  = ""
    @Volatile private var lastUrgentSpokeMs = 0L
    private val URGENT_COOLDOWN_MS = 3_000L
    // 5m 진입: 라벨이 5m 밖으로 나갔다 들어와야 재발화
    private val labels5mFired = mutableSetOf<String>()

    companion object {
        private const val FRAME_INTERVAL_MS = 500L  // 탐지 간격 (ms)
        // 경고 그룹 분류 (Detection.label = 한국어 그룹명)
        private val FULL_ALERT_GROUPS   = setOf("차량", "개인이동장치") // 5m 음성+진동 + 1m 긴급
        private val PERSON_ALERT_GROUPS = setOf("사람/동물")            // 1m 긴급 음성만
        private val CLOSE_ALERT_GROUPS  = setOf("고정장애물")           // 1m 긴급 음성만
        private val SIGNAL_ALERT_GROUPS = setOf("신호등/표지판")        // 1m 진동만
        // 위험도 점수 그룹 가중치
        private val GROUP_WEIGHT = mapOf(
            "차량"          to 1.0f,
            "개인이동장치"  to 0.9f,
            "사람/동물"     to 0.7f,
            "고정장애물"    to 0.5f,
            "신호등/표지판" to 0.3f,
        )
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
        detector       = ObjectDetector.create(this)
        remoteDetector = detector as? RemoteDetector
        segClient    = SegmentationClient(RemoteDetector.SERVER_URL)
        @Suppress("DEPRECATION")
        vibrator = getSystemService(VIBRATOR_SERVICE) as Vibrator
        vibCtrl  = VibrationController(vibrator)
        roadCtrl = RoadRepeatController(tts, Handler(Looper.getMainLooper()))

        setupButtons()
        setupVideoView()
    }

    override fun onPause() {
        super.onPause()
        pauseVideo()
    }

    override fun onDestroy() {
        stopFrameDetection()
        roadCtrl.stop()
        retriever?.release()
        tts.shutdown()
        runCatching { detector.close() }
        runCatching { segClient.close() }   // OkHttp 풀 정리
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
                    // 한 번만 JPEG 압축 → detect/segment 공유 (서버 캐시 명중)
                    val rd = remoteDetector
                    val sharedJpeg: ByteArray? = if (rd != null) {
                        val out = java.io.ByteArrayOutputStream()
                        bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG,
                            RemoteDetector.JPEG_QUALITY, out)
                        out.toByteArray()
                    } else null

                    val detections    = if (sharedJpeg != null) rd!!.detectJpeg(sharedJpeg)
                                        else detector.detect(bitmap, 0)
                    val serverMessage = remoteDetector?.lastMessage ?: ""
                    val dodgeDir      = remoteDetector?.lastDodge ?: "정면"

                    // 세그멘테이션 (2프레임마다)
                    frameCount++
                    val seg = if (frameCount % 2 == 0) {
                        if (sharedJpeg != null) segClient.segmentJpeg(sharedJpeg)
                        else segClient.segment(bitmap, 0)
                    } else null

                    runOnUiThread {
                        handleDetections(detections, serverMessage, dodgeDir)
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

    private fun handleDetections(detections: List<Detection>, serverMessage: String = "", dodgeDir: String = "정면") {
        // ── Depth 상태 추적 ──────────────────────────────────────────────────
        if (detections.isNotEmpty()) {
            if (detections.all { it.depthM == null }) {
                depthNullStreak++
                if (depthNullStreak >= 10 && !depthUnavailSpoken) {
                    depthUnavailSpoken = true
                    tts.speak("거리 측정을 사용할 수 없습니다. 장애물 거리를 알 수 없습니다.",
                        TextToSpeech.QUEUE_ADD, null, "depth-off")
                }
            } else {
                if (depthUnavailSpoken) {
                    depthUnavailSpoken = false
                    tts.speak("거리 측정이 활성화되었습니다.", TextToSpeech.QUEUE_ADD, null, "depth-on")
                }
                depthNullStreak = 0
            }
        }

        // 상태 텍스트
        detectionStatus.text = if (detections.isEmpty()) {
            "감지된 객체 없음"
        } else {
            val labels = detections.take(3).joinToString(", ") {
                "${it.label} ${(it.confidence * 100).toInt()}%"
            }
            "감지: $labels"
        }

        // 경고 대상: 기타 그룹만 제외
        val alertCandidates = detections.filter {
            it.label in FULL_ALERT_GROUPS || it.label in PERSON_ALERT_GROUPS ||
            it.label in CLOSE_ALERT_GROUPS || it.label in SIGNAL_ALERT_GROUPS
        }
        hasNearbyVehicle = alertCandidates.any { it.label in FULL_ALERT_GROUPS }

        // 5m 이내(또는 depth 모를 때 모두) 후보만 추림
        val nearby = alertCandidates.filter { it.depthM == null || it.depthM <= 5f }
        if (nearby.isEmpty()) {
            alert5mFired = false; alert1mFired = false; lastTrackedLabel = ""
            labels5mFired.clear()
            return
        }
        val nearbyLabels = nearby.map { it.label }.toSet()
        labels5mFired.retainAll(nearbyLabels)

        // 위험도 점수 = 면적 × 중심 가까움 × 신뢰도 × 그룹 가중치
        val worst = nearby.maxByOrNull { d ->
            val cw = 1f - kotlin.math.abs(d.centerX() - 0.5f)
            val gw = GROUP_WEIGHT[d.label] ?: 0.5f
            d.area() * (0.7f + 0.3f * cw) * d.confidence * gw
        } ?: return

        val depthM = worst.depthM
        val group  = worst.label

        // 다객체 분석
        val under1m = nearby.filter { it.depthM != null && it.depthM <= 1f }
        val under5m = nearby.filter { it.depthM != null && it.depthM in 1f..5f }
        val sameGroupCount = nearby.count { it.label == group }
        val multipleSameGroup = sameGroupCount >= 3
        val multipleUnder1m   = under1m.size >= 2
        val multipleUnder5m   = under5m.size >= 2

        if (worst.label != lastTrackedLabel) {
            alert5mFired = false; alert1mFired = false
            lastTrackedLabel = worst.label
        }

        val amplitude = when {
            depthM == null -> 180
            depthM <= 1f   -> 255
            depthM <= 3f   -> 180
            else           -> 100
        }

        // 신호등/표지판: 1m 진동만
        if (group in SIGNAL_ALERT_GROUPS) {
            if (depthM != null && depthM <= 1f) vibCtrl.vibrate(dodgeDir, amplitude)
            return
        }

        // 1m 이하 긴급 (같은 라벨 3초 쿨다운)
        if (depthM != null && depthM <= 1f) {
            val nowMs = SystemClock.elapsedRealtime()
            val sameLabel  = group == lastUrgentLabel
            val cooledDown = !sameLabel || (nowMs - lastUrgentSpokeMs >= URGENT_COOLDOWN_MS)
            if (!alert1mFired || cooledDown) {
                alert1mFired      = true
                lastUrgentLabel   = group
                lastUrgentSpokeMs = nowMs
                val noun = if (multipleSameGroup) "여러 ${groupNoun(group)}" else groupNoun(group)
                val urgentMsg = when {
                    serverMessage.isNotEmpty() && !multipleSameGroup -> serverMessage
                    else -> "위험! $dodgeDir $noun"
                }
                showWarning(urgentMsg, dodgeDir, amplitude, speakNow = true)
                if (multipleUnder1m) vibCtrl.vibrate(dodgeDir, 255)
                return
            }
            vibCtrl.vibrate(dodgeDir, amplitude)
            return
        }

        if (group in PERSON_ALERT_GROUPS || group in CLOSE_ALERT_GROUPS) return

        // 차량/개인이동장치 5m 음성 (라벨당 1회, 5m 밖 나갔다 들어와야 재발화)
        if (group !in labels5mFired) {
            labels5mFired += group
            alert5mFired = true
            val msg = when {
                multipleUnder5m ->
                    "여러 장애물 접근. 주의하세요"
                depthUnavailSpoken ->
                    "${dodgeDir}에 ${groupNoun(group)}"
                serverMessage.isNotEmpty() -> serverMessage
                else -> {
                    val distStr = if (depthM != null) "${String.format("%.1f", depthM)}m " else ""
                    "$dodgeDir $distStr${groupNoun(group)} 접근"
                }
            }
            showWarning(msg, dodgeDir, amplitude, speakNow = true)
            return
        }

        vibCtrl.vibrate(dodgeDir, amplitude)
    }

    private fun handleSegmentation(seg: SegmentResult) {
        val status            = seg.status
        val frontCls          = seg.frontCls
        val leftStatus        = seg.leftStatus
        val rightStatus       = seg.rightStatus
        val trafficLightColor = seg.trafficLightColor

        // ── 신호등 색상 안내 (횡단보도 위에서만) ─────────────────────────
        if (status == "crosswalk" && trafficLightColor != lastTrafficLightColor) {
            lastTrafficLightColor = trafficLightColor
            when (trafficLightColor) {
                "green" -> tts.speak("초록불입니다. 건너세요.",         TextToSpeech.QUEUE_FLUSH, null, "light-green")
                "red"   -> tts.speak("빨간불입니다. 멈추세요.",         TextToSpeech.QUEUE_FLUSH, null, "light-red")
                ""      -> {
                    tts.speak("신호등을 확인할 수 없습니다. 주변을 살펴보세요.",
                        TextToSpeech.QUEUE_ADD, null, "light-unknown")
                    if (!hasNearbyVehicle) {
                        tts.speak("건너셔도 됩니다.", TextToSpeech.QUEUE_ADD, null, "light-safe")
                    }
                }
            }
        }
        if (status != "crosswalk") lastTrafficLightColor = "unset"

        val surfaceKey = if (frontCls.isNotEmpty()) frontCls else status
        if (surfaceKey == lastSurfaceStatus) return
        lastSurfaceStatus = surfaceKey

        roadCtrl.stop()

        val message = surfaceMessageFor(frontCls, status)
        if (message != null) {
            val qMode = if (status == "crosswalk") TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
            tts.speak(message, qMode, null, "surface-${SystemClock.elapsedRealtime()}")
        }

        if (status == "road" || frontCls == "bike_lane") {
            val situation = if (frontCls == "bike_lane") "자전거 도로입니다." else "차도입니다."
            roadCtrl.start(situation, leftStatus, rightStatus)
        }
    }

    private fun showWarning(message: String, side: String, amplitude: Int = 180, speakNow: Boolean = false) {
        warningText.text = message
        warningBanner.visibility = View.VISIBLE

        if (speakNow) {
            tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, "warn-${SystemClock.elapsedRealtime()}")
        }

        vibCtrl.vibrate(side, amplitude)
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
