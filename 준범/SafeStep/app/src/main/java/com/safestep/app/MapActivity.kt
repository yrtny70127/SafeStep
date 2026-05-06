package com.safestep.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.safestep.app.detect.Detection
import com.safestep.app.detect.ObjectDetector
import com.safestep.app.detect.RemoteDetector
import com.safestep.app.detect.SegmentationClient
import com.safestep.app.detect.SignalClient
import com.safestep.app.navigation.NavigationGuide
import com.safestep.app.navigation.PoiResult
import com.safestep.app.navigation.RouteResult
import com.safestep.app.navigation.TmapService
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline
import org.osmdroid.views.overlay.mylocation.GpsMyLocationProvider
import org.osmdroid.views.overlay.mylocation.MyLocationNewOverlay
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MapActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    // ── Views ─────────────────────────────────────────────────────────────────
    private lateinit var mapView: MapView
    private lateinit var previewView: PreviewView
    private lateinit var bboxOverlay: BoundingBoxOverlay
    private lateinit var segmentOverlay: ImageView
    private lateinit var destinationText: TextView
    private lateinit var destMicButton: Button
    private lateinit var warningBanner: LinearLayout
    private lateinit var warningText: TextView
    private lateinit var navArrow: TextView
    private lateinit var navInstruction: TextView
    private lateinit var navDistance: TextView
    private lateinit var navStepCount: TextView
    private lateinit var backToModeButton: Button

    // ── Core ──────────────────────────────────────────────────────────────────
    private lateinit var tts: TextToSpeech
    private lateinit var cameraExecutor: ExecutorService
    /** 배너 자동 숨김용 — removeCallbacks로 항상 리셋해 마지막 경고 기준 3초 유지 */
    private val mainHandler       = Handler(Looper.getMainLooper())
    private val hideBannerRunnable = Runnable { warningBanner.visibility = View.GONE }
    private lateinit var detectExecutor: ExecutorService // 탐지 전용 스레드 (cameraExecutor 블로킹 방지)
    private lateinit var segExecutor: ExecutorService    // 세그멘테이션 전용 스레드
    private lateinit var signalExecutor: ExecutorService // 신호등 전용 스레드
    private lateinit var fastExecutor: ExecutorService   // Fast Lane 전용 스레드

    /**
     * "skip-if-busy" 플래그 — 이전 요청이 아직 처리 중이면 새 프레임을 버림.
     * 서버 응답 속도에 자동 맞춰져 큐 적체 없이 항상 최신 프레임만 전송.
     */
    private val detectBusy = java.util.concurrent.atomic.AtomicBoolean(false)
    private val segBusy    = java.util.concurrent.atomic.AtomicBoolean(false)
    private val signalBusy = java.util.concurrent.atomic.AtomicBoolean(false)
    private val fastBusy   = java.util.concurrent.atomic.AtomicBoolean(false)  // Fast Lane 전용
    private var vibrator: Vibrator? = null
    private lateinit var detector: ObjectDetector
    private lateinit var segClient: SegmentationClient
    private lateinit var signalClient: SignalClient

    // ── Location ──────────────────────────────────────────────────────────────
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var locationCallback: LocationCallback
    private var currentLocation: Location? = null

    // ── Compass (Heading-up) ──────────────────────────────────────────────────
    private lateinit var sensorManager: SensorManager
    private var rotationSensor: Sensor? = null
    private var displayAzimuth = 0f   // 저역통과 필터 적용된 부드러운 방위각 (지도 회전 + 전방 오프셋 공용)
    private val rotMatrix  = FloatArray(9)
    private val orientVals = FloatArray(3)

    // ── Map overlays ──────────────────────────────────────────────────────────
    private var myLocationOverlay: MyLocationNewOverlay? = null
    private var routePolyline: Polyline? = null
    private var destMarker: Marker? = null

    // ── Navigation ────────────────────────────────────────────────────────────
    private lateinit var navGuide: NavigationGuide
    private var totalSteps = 0
    /** 전체 경로 포인트 (잔여 경로 업데이트 + 이탈 감지용) */
    private var allPathPoints: List<org.osmdroid.util.GeoPoint> = emptyList()
    /** 재탐색용 목적지 저장 */
    private var currentDest: PoiResult? = null
    /** 경로 이탈 경고 마지막 시각 */
    @Volatile private var lastOffRouteWarnMs = 0L

    // ── Speech ────────────────────────────────────────────────────────────────
    private var speechRecognizer: SpeechRecognizer? = null
    @Volatile private var isListening = false   // cameraExecutor 에서도 읽힘 → Volatile 필요

    // ── Segmentation ──────────────────────────────────────────────────────────
    @Volatile private var frameCount = 0
    @Volatile private var lastSurfaceStatus = ""

    // ── Signal (신호등) ────────────────────────────────────────────────────────
    @Volatile private var lastTrafficLightStatus: String? = "init"  // 최초 발화 방지
    /** null 복귀 시 "건너셔도 됩니다" 오발화 방지용 — 마지막 실제 감지된 신호 색 */
    @Volatile private var lastKnownSignalColor: String? = null

    // ── Surface zones (3구역 회피 방향) ──────────────────────────────────────
    @Volatile private var lastZones: Map<String, String> = emptyMap()

    // ── Area-growth 접근 속도 (서버 depth 없을 때 fallback) ───────────────
    // label → deque of (area, elapsedMs)
    private val areaHistory = mutableMapOf<String, ArrayDeque<Pair<Float, Long>>>()

    // ── 서버 연결 상태 ─────────────────────────────────────────────────────
    @Volatile private var serverConnected    = false
    @Volatile private var lastServerCheckMs  = 0L

    // ── 횡단보도 + 위험 노면 반복 ─────────────────────────────────────────
    @Volatile private var onCrosswalk         = false
    @Volatile private var lastSurfaceRepeatMs = 0L

    // ── 반대 방향 감지 ────────────────────────────────────────────────────
    private var prevNavLocation: Location?    = null   // 이전 GPS 위치 (방위각 계산용)
    @Volatile private var lastWrongWayWarnMs  = 0L
    @Volatile private var lastStepChangeMs    = 0L     // 스텝 바뀐 직후 오감지 방지

    // ── 횡단보도 × 신호등 없음 → 차량 감지 여부 ─────────────────────────
    @Volatile private var lastVehicleDetectMs = 0L
    @Volatile private var vehicleUrgentUntilMs = 0L  // Fast Lane: 차량 긴급 시 seg/signal 중단

    // ── Detection ─────────────────────────────────────────────────────────────
    @Volatile private var lastSpoken = ""
    @Volatile private var lastSpeakMs = 0L
    /** 보행 모드: 앱 시작부터 장애물 TTS 활성화 (목적지 없어도 경고) */
    @Volatile private var detectionTtsEnabled = true
    /** 가속도계 기반 이동 여부 — false 이면 배터리 절약 모드 */
    @Volatile private var isMoving = true
    private var stationaryCount = 0
    private var linearAccelSensor: Sensor? = null

    // ── 경고 레벨 / 장애물 그룹 / 거리 단계 열거형 ────────────────────────────
    private enum class WarnLevel { URGENT, VIB_ONLY, NORMAL, NONE }
    private enum class ObjGroup  { VEHICLE, MICRO, PERSON, FIXED, SIGNAL, OTHER }
    private enum class DistTier  { CLOSE, MEDIUM, FAR, NONE }
    //                              ≤1.5m   ≤3m    ≤5m

    companion object {
        private const val TAG               = "MapActivity"
        private const val REQ_PERM          = 200
        private const val DANGER_AREA       = 0.20f
        private const val VERY_CLOSE_AREA   = 0.40f
        private const val SPEAK_COOLDOWN_MS = 2000L
        /** 위험 노면(차도/자전거도로) 머무는 동안 방향 유도 반복 간격 */
        private const val SURFACE_REPEAT_MS = 5_000L

        private val PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.RECORD_AUDIO
        )
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Lifecycle
    // ══════════════════════════════════════════════════════════════════════════

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Configuration.getInstance().load(this, getSharedPreferences("osmdroid", 0))
        setContentView(R.layout.activity_map)

        mapView         = findViewById(R.id.mapView)
        previewView     = findViewById(R.id.previewView)
        bboxOverlay     = findViewById(R.id.bboxOverlay)
        segmentOverlay  = findViewById(R.id.segmentOverlay)
        destinationText = findViewById(R.id.destinationText)
        destMicButton   = findViewById(R.id.destMicButton)
        warningBanner   = findViewById(R.id.warningBanner)
        warningText     = findViewById(R.id.warningText)
        navArrow        = findViewById(R.id.navArrow)
        navInstruction  = findViewById(R.id.navInstruction)
        navDistance     = findViewById(R.id.navDistance)
        navStepCount    = findViewById(R.id.navStepCount)
        backToModeButton = findViewById(R.id.backToModeButton)

        backToModeButton.setOnClickListener {
            startActivity(Intent(this, SplashActivity::class.java))
            finish()
        }

        tts            = TextToSpeech(this, this)
        @Suppress("DEPRECATION")
        vibrator       = getSystemService(VIBRATOR_SERVICE) as Vibrator
        cameraExecutor = Executors.newSingleThreadExecutor()
        detectExecutor = Executors.newSingleThreadExecutor()
        segExecutor    = Executors.newSingleThreadExecutor()
        signalExecutor = Executors.newSingleThreadExecutor()
        fastExecutor   = Executors.newSingleThreadExecutor()
        // SharedPreferences에서 서버 URL 불러와 적용
        val savedUrl = getSharedPreferences(SplashActivity.PREFS_NAME, MODE_PRIVATE)
            .getString(SplashActivity.KEY_SERVER_URL, SplashActivity.DEFAULT_SERVER_URL)
        if (!savedUrl.isNullOrEmpty()) RemoteDetector.SERVER_URL = savedUrl

        detector       = ObjectDetector.create(this)
        segClient      = SegmentationClient(RemoteDetector.SERVER_URL)
        signalClient   = SignalClient(RemoteDetector.SERVER_URL)
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)

        // 나침반 + 이동 감지 센서
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        rotationSensor    = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        linearAccelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)

        navGuide = NavigationGuide(tts).apply {
            onStepChanged = { step ->
                lastStepChangeMs = SystemClock.elapsedRealtime()  // 반대방향 오감지 방지
                prevNavLocation  = null                            // 이전 위치 초기화
                runOnUiThread {
                    updateNavUI(step.description, step.distance)
                    navStepCount.text = "${navGuide.currentStepNumber()}/$totalSteps"
                }
            }
            onArrived = { runOnUiThread { onArrived() } }
        }

        setupMap()
        setupMicButton()
        buildLocationCallback()

        if (hasAllPermissions()) startAll()
        else ActivityCompat.requestPermissions(this, PERMISSIONS, REQ_PERM)
    }

    override fun onResume() {
        super.onResume()
        mapView.onResume()
        rotationSensor?.let {
            sensorManager.registerListener(compassListener, it, SensorManager.SENSOR_DELAY_UI)
        }
        linearAccelSensor?.let {
            sensorManager.registerListener(motionListener, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
    }

    override fun onPause() {
        super.onPause()
        mapView.onPause()
        sensorManager.unregisterListener(compassListener)
        sensorManager.unregisterListener(motionListener)
    }

    override fun onDestroy() {
        // 대기 중인 배너 숨김 콜백 취소 — Activity 참조 메모리 누수 방지
        mainHandler.removeCallbacks(hideBannerRunnable)
        mapView.onDetach()
        cameraExecutor.shutdown()
        detectExecutor.shutdown()
        segExecutor.shutdown()
        signalExecutor.shutdown()
        fastExecutor.shutdown()
        tts.shutdown()
        speechRecognizer?.destroy()
        if (::fusedLocationClient.isInitialized && ::locationCallback.isInitialized)
            fusedLocationClient.removeLocationUpdates(locationCallback)
        runCatching { detector.close() }
        super.onDestroy()
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 나침반 → Heading-up 지도 회전
    // ══════════════════════════════════════════════════════════════════════════

    private val compassListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            SensorManager.getRotationMatrixFromVector(rotMatrix, event.values)
            SensorManager.getOrientation(rotMatrix, orientVals)
            val az = ((Math.toDegrees(orientVals[0].toDouble()).toFloat()) + 360f) % 360f

            // 저역통과 필터 (alpha=0.12): 떨림 제거하면서 빠른 회전도 따라감
            // 0/360 경계 처리: 차이를 -180~180 범위로 정규화
            var diff = az - displayAzimuth
            if (diff > 180f)  diff -= 360f
            if (diff < -180f) diff += 360f
            displayAzimuth = (displayAzimuth + 0.12f * diff + 360f) % 360f

            mapView.setMapOrientation(-displayAzimuth)
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    /** 선형 가속도 기반 이동/정지 판단 — 정지 시 detection 주기 줄여 배터리 절약 */
    private val motionListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            val mag = Math.sqrt(
                (event.values[0] * event.values[0] +
                 event.values[1] * event.values[1] +
                 event.values[2] * event.values[2]).toDouble()
            ).toFloat()
            if (mag > 0.6f) {          // 움직임 감지
                stationaryCount = 0
                isMoving = true
            } else if (++stationaryCount > 20) {  // 20샘플 연속 정지 (~2초)
                isMoving = false
            }
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Permissions
    // ══════════════════════════════════════════════════════════════════════════

    private fun hasAllPermissions() = PERMISSIONS.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_PERM && hasAllPermissions()) startAll()
    }

    private fun startAll() {
        startLocationUpdates()
        startCamera()
        setupSpeechRecognizer()
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Map
    // ══════════════════════════════════════════════════════════════════════════

    private fun setupMap() {
        mapView.setTileSource(TileSourceFactory.MAPNIK)
        mapView.setMultiTouchControls(true)
        mapView.controller.setZoom(18.0)

        // 내 위치 표시 (자동 팔로우는 위치 콜백에서 직접 처리)
        myLocationOverlay = MyLocationNewOverlay(GpsMyLocationProvider(this), mapView).apply {
            enableMyLocation()
        }
        mapView.overlays.add(myLocationOverlay)
    }

    private fun showRouteOnMap(result: RouteResult, dest: PoiResult) {
        allPathPoints = result.pathPoints   // 전체 경로 저장 (잔여선 + 이탈 감지)

        routePolyline?.let { mapView.overlays.remove(it) }
        destMarker?.let    { mapView.overlays.remove(it) }

        routePolyline = Polyline(mapView).apply {
            setPoints(result.pathPoints)
            outlinePaint.color       = Color.parseColor("#F97316")
            outlinePaint.strokeWidth = 10f
        }
        mapView.overlays.add(routePolyline)

        destMarker = Marker(mapView).apply {
            position = GeoPoint(dest.lat, dest.lon)
            title    = dest.name
            setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)
        }
        mapView.overlays.add(destMarker)
        mapView.invalidate()
    }

    /**
     * 사용자 위치를 지도에 부드럽게 고정.
     * 내비 중에는 진행 방향으로 40m 앞을 중심에 놓아
     * 사용자가 화면 하단 1/3 위치에 오도록 함 → 전방 시야 확보.
     */
    private fun centerMapOnUser(loc: Location) {
        val centerPoint = if (navGuide.isRunning()) {
            // 필터 적용된 방위각으로 40m 오프셋 (지도 회전과 동일한 값 사용 → 부드러움 일관성)
            val offsetM = 40.0
            val headingRad = Math.toRadians(displayAzimuth.toDouble())
            val dLat = offsetM * kotlin.math.cos(headingRad) / 111320.0
            val dLon = offsetM * kotlin.math.sin(headingRad) /
                    (111320.0 * kotlin.math.cos(Math.toRadians(loc.latitude)))
            GeoPoint(loc.latitude + dLat, loc.longitude + dLon)
        } else {
            GeoPoint(loc.latitude, loc.longitude)
        }
        mapView.controller.animateTo(centerPoint)
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Location
    // ══════════════════════════════════════════════════════════════════════════

    private fun buildLocationCallback() {
        locationCallback = object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                val loc = result.lastLocation ?: return
                currentLocation = loc
                navGuide.updateLocation(loc.latitude, loc.longitude)

                // 지도 중앙 고정
                centerMapOnUser(loc)

                if (navGuide.isRunning()) {
                    val dist = navGuide.distToCurrentStep(loc.latitude, loc.longitude)
                    val zoom = when {
                        dist <= 30  -> 20.0
                        dist <= 80  -> 19.0
                        dist <= 200 -> 18.0
                        else        -> 17.0
                    }
                    mapView.controller.setZoom(zoom)

                    // 거리 UI 실시간 갱신
                    val step = navGuide.currentStep()
                    runOnUiThread {
                        navDistance.text = formatDist(dist)
                        if (step != null) navInstruction.text = step.description
                    }

                    // 잔여 경로선 업데이트
                    updateRemainingRoute(loc.latitude, loc.longitude)

                    // 경로 이탈 감지
                    checkOffRoute(loc.latitude, loc.longitude)

                    // 반대 방향 감지
                    checkWrongWay(loc)
                }
            }
        }
    }

    private fun startLocationUpdates() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED) return
        val req = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1_000L)
            .setMinUpdateDistanceMeters(1f)
            .build()
        fusedLocationClient.requestLocationUpdates(req, locationCallback, mainLooper)
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Navigation UI
    // ══════════════════════════════════════════════════════════════════════════

    private fun updateNavUI(instruction: String, distanceM: Int) {
        navInstruction.text = instruction
        navDistance.text    = formatDist(distanceM)
        navArrow.text       = directionArrow(instruction)
    }

    private fun formatDist(m: Int): String = when {
        m <= 0    -> ""
        m < 1000  -> "${m}m"
        else      -> "${"%.1f".format(m / 1000.0)}km"
    }

    private fun directionArrow(instruction: String): String = when {
        instruction.contains("좌회전")          -> "↰"
        instruction.contains("우회전")          -> "↱"
        instruction.contains("좌측")            -> "↖"
        instruction.contains("우측")            -> "↗"
        instruction.contains("유턴")            -> "↩"
        instruction.contains("목적지") ||
                instruction.contains("도착")   -> "🏁"
        instruction.contains("출발")            -> "↑"
        else                                    -> "↑"
    }

    private fun onArrived() {
        navArrow.text        = "🏁"
        navInstruction.text  = "목적지 도착"
        navDistance.text     = ""
        navStepCount.text    = ""
        destinationText.text = "목적지를 말씀해주세요"
        mapView.controller.setZoom(18.0)
        // 도착 후 새 목적지 탐색 시 "새로운 경로" 오발화 방지
        allPathPoints = emptyList()
        currentDest   = null
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Voice — 목적지 입력
    // ══════════════════════════════════════════════════════════════════════════

    private fun setupMicButton() {
        destMicButton.setOnClickListener {
            if (isListening) stopListening() else startListening()
        }
    }

    private fun setupSpeechRecognizer() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            destMicButton.isEnabled = false
            return
        }
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this).apply {
            setRecognitionListener(destinationRecognitionListener)
        }
    }

    private fun startListening() {
        isListening = true
        destMicButton.text = "⏹"
        tts.speak("목적지를 말씀해주세요.", TextToSpeech.QUEUE_FLUSH, null, "dest-prompt")
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
        }
        speechRecognizer?.startListening(intent)
    }

    private fun stopListening() {
        isListening = false
        destMicButton.text = "🎙"
        speechRecognizer?.stopListening()
    }

    private fun handleDestinationQuery(query: String) {
        destinationText.text = "\"$query\" 검색 중…"
        // 목적지 인식 시점부터 장애물 경고 TTS 활성화
        detectionTtsEnabled = true
        tts.speak("$query 검색합니다.", TextToSpeech.QUEUE_FLUSH, null, "searching")

        TmapService.searchPoi(query) { pois ->
            // TmapService는 main.post로 콜백하므로 이미 메인 스레드
            // → Activity 소멸 후 TTS/View 접근 방지
            runOnUiThread {
                if (isDestroyed) return@runOnUiThread
                if (pois.isEmpty()) {
                    tts.speak("검색 결과가 없습니다. 다시 말씀해주세요.", TextToSpeech.QUEUE_FLUSH, null, "no-poi")
                    destinationText.text = "목적지를 말씀해주세요"
                    return@runOnUiThread
                }
                val dest = pois.first()
                destinationText.text = dest.name
                tts.speak("${dest.name}(으)로 경로를 탐색합니다.", TextToSpeech.QUEUE_FLUSH, null, "found")

                val loc = currentLocation
                if (loc == null) {
                    tts.speak("현재 위치를 가져오는 중입니다. 잠시 후 다시 시도해주세요.",
                        TextToSpeech.QUEUE_ADD, null, "no-loc")
                    return@runOnUiThread
                }
                fetchRoute(loc.latitude, loc.longitude, dest)
            }
        }
    }

    private fun fetchRoute(sLat: Double, sLon: Double, dest: PoiResult) {
        currentDest = dest   // 재탐색용 저장
        TmapService.searchPedestrianRoute(sLat, sLon, dest.lat, dest.lon, dest.name) { result ->
            if (result == null || result.steps.isEmpty()) {
                // 성공 경로와 동일하게 메인 스레드에서 처리 — Activity 소멸 후 TTS 접근 방지
                runOnUiThread {
                    if (isDestroyed) return@runOnUiThread
                    tts.speak("경로를 찾을 수 없습니다.", TextToSpeech.QUEUE_FLUSH, null, "no-route")
                }
                return@searchPedestrianRoute
            }
            // ── 결과 처리 전체를 메인 스레드에서 실행 ────────────────────────────
            // 이유 ①: showRouteOnMap()이 mapView.overlays를 수정 → osmdroid는 메인 스레드 필요
            // 이유 ②: navGuide.start()와 메인 스레드의 navGuide.updateLocation()이 동시에
            //         실행되면 steps/currentIdx 레이스 컨디션 발생
            runOnUiThread {
                // 재탐색인지 최초 탐색인지 구분 — showRouteOnMap() 전에 체크해야 함
                val intro = if (allPathPoints.isNotEmpty()) "새로운 경로를 안내합니다."
                            else "경로 탐색 완료."

                totalSteps = result.steps.size
                showRouteOnMap(result, dest)

                val dist = if (result.totalDistanceM >= 1000)
                    "${"%.1f".format(result.totalDistanceM / 1000.0)}km"
                else "${result.totalDistanceM}m"
                val min = result.totalTimeSec / 60

                // ① 경로 요약 먼저 큐에 등록
                tts.speak(
                    "$intro 총 $dist, 약 ${min}분 소요됩니다.",
                    TextToSpeech.QUEUE_ADD, null, "route-ready"
                )

                // ② start() 내부의 announce(0)이 첫 지시문을 요약 다음에 큐에 추가됨
                //    → 사용자는 "경로 탐색 완료. 총 …" 다음 "직진하세요" 순서로 듣게 됨
                navGuide.start(result.steps)

                // 첫 안내 UI 표시
                val first = result.steps.first()
                updateNavUI(first.description, first.distance)
                navStepCount.text = "1/$totalSteps"
            }
        }
    }

    private val destinationRecognitionListener = object : RecognitionListener {
        override fun onResults(results: Bundle?) {
            isListening = false
            destMicButton.text = "🎙"
            val text = results
                ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                ?.firstOrNull() ?: return
            handleDestinationQuery(text)
        }
        override fun onError(error: Int) {
            isListening = false
            destMicButton.text = "🎙"
            tts.speak("음성 인식에 실패했습니다. 다시 눌러주세요.", TextToSpeech.QUEUE_FLUSH, null, "rec-err")
        }
        override fun onReadyForSpeech(p: Bundle?) {}
        override fun onBeginningOfSpeech() {}
        override fun onRmsChanged(v: Float) {}
        override fun onBufferReceived(b: ByteArray?) {}
        override fun onEndOfSpeech() {}
        override fun onPartialResults(p: Bundle?) {}
        override fun onEvent(t: Int, p: Bundle?) {}
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Camera
    // ══════════════════════════════════════════════════════════════════════════

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview  = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(cameraExecutor, ::analyzeFrame) }
            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            } catch (e: Exception) {
                Log.e(TAG, "카메라 바인딩 실패", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @OptIn(ExperimentalGetImage::class)
    private fun analyzeFrame(imageProxy: ImageProxy) {
        try {
            frameCount++

            // 배터리 절약: 정지 중이면 3프레임에 1번만 처리
            if (!isMoving && frameCount % 3 != 0) {
                imageProxy.close()
                return
            }

            val bitmap   = imageProxy.toBitmap()
            val rotation = imageProxy.imageInfo.rotationDegrees

            // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            // 회전 보정 1회 공유:
            //   세 클라이언트가 각자 회전하던 것을 카메라 스레드에서 한 번만 수행.
            //   이후 각 클라이언트는 rotationDegrees=0 으로 호출 → 회전 생략.
            //   rotation==0 이면 imageProxy 반환 후 bitmap이 무효화되므로 copy().
            // skip-if-busy 전략:
            //   이전 요청이 아직 처리 중이면 현재 프레임을 버림.
            // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            val corrected: android.graphics.Bitmap = if (rotation != 0) {
                val m = android.graphics.Matrix().apply { postRotate(rotation.toFloat()) }
                android.graphics.Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
            } else {
                bitmap.copy(bitmap.config ?: android.graphics.Bitmap.Config.ARGB_8888, false)
            }

            // 🚀 Fast Lane — 매 프레임, 차량·사람 전용 (imgsz=160)
            //    일반 detect 보다 ~2× 빠른 응답으로 긴급 충돌 경고 즉시 발화
            if (fastBusy.compareAndSet(false, true)) {
                val fastDet = detector as? RemoteDetector
                if (fastDet != null) {
                    fastExecutor.execute {
                        try {
                            val detections = fastDet.detectFast(corrected, 0)
                            if (detections.isNotEmpty()) handleDetections(detections)
                        } catch (e: Exception) {
                            Log.w(TAG, "고속 탐지 실패: ${e.message}")
                        } finally {
                            fastBusy.set(false)
                        }
                    }
                } else {
                    fastBusy.set(false)
                }
            }

            // ① 일반 탐지 — 2프레임마다, 모든 객체 + depth (Fast Lane 보완)
            if (frameCount % 2 == 0 && detectBusy.compareAndSet(false, true)) {
                detectExecutor.execute {
                    try {
                        val detections = detector.detect(corrected, 0)
                        handleDetections(detections)
                    } catch (e: Exception) {
                        Log.w(TAG, "탐지 실패: ${e.message}")
                    } finally {
                        detectBusy.set(false)
                    }
                }
            }

            // ④ 서버 연결 30초 주기 확인
            if (frameCount % 300 == 0) {
                val now = SystemClock.elapsedRealtime()
                if (now - lastServerCheckMs >= 30_000L) {
                    if (!serverConnected) repeatServerOfflineWarning()
                    checkServerHealth()
                }
            }

            // ② 세그멘테이션 — 2프레임마다, 차량 긴급 접근 중엔 완전 중단
            val vehicleUrgent = SystemClock.elapsedRealtime() < vehicleUrgentUntilMs
            if (!vehicleUrgent && frameCount % 2 == 0 && segBusy.compareAndSet(false, true)) {
                segExecutor.execute {
                    try {
                        val seg = segClient.segment(corrected, 0)
                        if (seg != null) handleSegmentation(seg)
                    } catch (e: Exception) {
                        Log.w(TAG, "세그멘테이션 실패: ${e.message}")
                    } finally {
                        segBusy.set(false)
                    }
                }
            }

            // ③ 신호등 — 3프레임마다, 차량 긴급 접근 중엔 완전 중단
            if (!vehicleUrgent && frameCount % 3 == 0 && signalBusy.compareAndSet(false, true)) {
                signalExecutor.execute {
                    try {
                        val signal = signalClient.detect(corrected, 0)
                        handleTrafficLight(signal?.color)
                    } catch (e: Exception) {
                        Log.w(TAG, "신호등 감지 실패: ${e.message}")
                    } finally {
                        signalBusy.set(false)
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "프레임 분석 실패", e)
        } finally {
            imageProxy.close()
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 탐지 결과 처리
    // ══════════════════════════════════════════════════════════════════════════

    private fun handleDetections(detections: List<Detection>) {
        bboxOverlay.updateDetections(detections)

        val nowMs = SystemClock.elapsedRealtime()
        areaHistory.entries.removeAll { (_, h) -> h.isEmpty() || nowMs - h.last().second > 5_000L }
        if (detections.isEmpty()) return

        // ── 탐지별 경고 후보 계산 ─────────────────────────────────────────────
        data class Candidate(val det: Detection, val level: WarnLevel, val amp: Int, val side: String)

        val candidates = detections.mapNotNull { det ->
            val tier = distTier(det)
            val grp  = obstacleGroup(det)
            val (lvl, amp) = warnLevelFor(grp, tier)
            if (lvl == WarnLevel.NONE) return@mapNotNull null
            val cx = det.centerX()
            val side = when { cx < 0.33f -> "왼쪽"; cx > 0.66f -> "오른쪽"; else -> "앞" }
            Candidate(det, lvl, amp, side)
        }
        if (candidates.isEmpty()) return

        // 횡단보도 신호 없음 판단용: 차량/개인이동장치가 근처에 있으면 시각 기록
        val hasNearbyVehicle = candidates.any { c ->
            val g = obstacleGroup(c.det)
            (g == ObjGroup.VEHICLE || g == ObjGroup.MICRO) && c.level != WarnLevel.NONE
        }
        if (hasNearbyVehicle) {
            lastVehicleDetectMs = nowMs
            // Fast Lane: URGENT 차량이면 다음 1.5초간 seg/signal 중단 → detect 전용 서버 자원 확보
            val isUrgent = candidates.any { c ->
                val g = obstacleGroup(c.det)
                (g == ObjGroup.VEHICLE || g == ObjGroup.MICRO) && c.level == WarnLevel.URGENT
            }
            if (isUrgent) vehicleUrgentUntilMs = nowMs + 1500L
        }

        // 가장 높은 우선순위 선택 (URGENT > VIB_ONLY > NORMAL)
        val top = candidates.minByOrNull { it.level.ordinal } ?: return
        val det = top.det

        // ── UI 배너 ───────────────────────────────────────────────────────────
        val distStr    = det.depthM?.let { "${"%.1f".format(it)}m " } ?: ""
        val displayMsg = "${top.side} $distStr${det.label}"
        runOnUiThread {
            warningText.text = displayMsg
            warningBanner.visibility = View.VISIBLE
        }
        // 이전 숨김 콜백을 취소하고 마지막 경고 기준으로 3초 타이머를 리셋
        mainHandler.removeCallbacks(hideBannerRunnable)
        mainHandler.postDelayed(hideBannerRunnable, 3_000)

        if (!detectionTtsEnabled || isListening) return

        // ── 경고 레벨별 처리 ─────────────────────────────────────────────────
        when (top.level) {
            WarnLevel.URGENT -> {
                // 1m 이하: 쿨다운 무시, FLUSH로 즉시 발화
                val msg = "위험! ${top.side} ${det.label} 매우 가깝습니다"
                vibrateForDirection(top.side, top.amp)
                tts.speak(msg, TextToSpeech.QUEUE_FLUSH, null, "urg-$nowMs")
                lastSpoken  = msg
                lastSpeakMs = nowMs
            }
            WarnLevel.VIB_ONLY -> {
                // 3m: 진동만, 쿨다운 적용
                if (nowMs - lastSpeakMs >= SPEAK_COOLDOWN_MS) {
                    lastSpeakMs = nowMs
                    vibrateForDirection(top.side, top.amp)
                }
            }
            WarnLevel.NORMAL -> {
                // 5m: 음성 + 약한 진동, 쿨다운 적용
                val dodge      = buildDodgeHint(top.side)
                val dodgeShort = dodgeHintShort(dodge)
                val spokenMsg  = if (dodgeShort != null) "${top.side} ${det.label}, $dodgeShort"
                                 else "${top.side} ${det.label}"
                if (spokenMsg != lastSpoken || nowMs - lastSpeakMs >= SPEAK_COOLDOWN_MS) {
                    lastSpoken  = spokenMsg
                    lastSpeakMs = nowMs
                    vibrateForDirection(top.side, top.amp)
                    tts.speak(spokenMsg, TextToSpeech.QUEUE_ADD, null, "norm-$nowMs")
                }
            }
            WarnLevel.NONE -> { /* unreachable */ }
        }
    }

    // ── 장애물 분류 헬퍼 ──────────────────────────────────────────────────────

    private fun obstacleGroup(det: Detection): ObjGroup = when (det.group) {
        "vehicle" -> ObjGroup.VEHICLE
        "micro"   -> ObjGroup.MICRO
        "person"  -> ObjGroup.PERSON
        null -> {
            val lbl = det.label.lowercase()
            if (lbl.contains("traffic") || lbl.contains("light") || lbl.contains("sign"))
                ObjGroup.SIGNAL
            else
                ObjGroup.FIXED
        }
        else -> ObjGroup.OTHER
    }

    /** depth + 접근속도 + 면적을 종합해 거리 단계 결정 */
    private fun distTier(det: Detection): DistTier {
        val depth = det.depthM
        if (depth != null) {
            return when {
                depth <= 1.5f -> DistTier.CLOSE
                depth <= 3f   -> DistTier.MEDIUM
                depth <= 5f   -> DistTier.FAR
                else          -> DistTier.NONE
            }
        }
        // depth 없음 → 접근 속도 + 면적 fallback
        val speedD = det.approachSpeed
        val speedA = areaGrowthRate(det.label, det.area())
        return when {
            speedD != null && speedD >= 1.0f  -> DistTier.MEDIUM
            speedA != null && speedA >= 0.05f -> DistTier.MEDIUM
            det.area() >= VERY_CLOSE_AREA     -> DistTier.MEDIUM
            speedD != null && speedD >= 0.3f  -> DistTier.FAR
            speedA != null && speedA >= 0.02f -> DistTier.FAR
            det.area() >= DANGER_AREA         -> DistTier.FAR
            else                              -> DistTier.NONE
        }
    }

    /** 그룹 × 거리 단계 → (경고 레벨, 진동 세기) */
    private fun warnLevelFor(grp: ObjGroup, tier: DistTier): Pair<WarnLevel, Int> = when (grp) {
        ObjGroup.VEHICLE, ObjGroup.MICRO -> when (tier) {
            DistTier.CLOSE  -> WarnLevel.URGENT   to 255  // 음성(긴급) + 진동(최대)
            DistTier.MEDIUM -> WarnLevel.VIB_ONLY to 180  // 진동(중)만
            DistTier.FAR    -> WarnLevel.NORMAL   to 100  // 음성 1회 + 진동(약)
            DistTier.NONE   -> WarnLevel.NONE     to 0
        }
        ObjGroup.PERSON, ObjGroup.FIXED -> when (tier) {
            DistTier.CLOSE  -> WarnLevel.URGENT to 200   // 1m 이하: 긴급 음성만
            else            -> WarnLevel.NONE   to 0
        }
        ObjGroup.SIGNAL -> when (tier) {
            DistTier.CLOSE  -> WarnLevel.VIB_ONLY to 255 // 1m 이하: 진동(최대)만
            else            -> WarnLevel.NONE     to 0
        }
        ObjGroup.OTHER -> WarnLevel.NONE to 0
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 신호등 상태 처리 — 횡단보도 여부에 따라 메시지 분기
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    private fun handleTrafficLight(color: String?) {
        if (color == lastTrafficLightStatus) return
        lastTrafficLightStatus = color
        if (color != null) lastKnownSignalColor = color   // 실제 감지된 색만 기록

        val msg = if (onCrosswalk) {
            // 횡단보도 위: 인식 불가도 안내
            when (color) {
                "red"      -> "빨간불입니다. 멈추세요."
                "green"    -> "초록불입니다. 건너세요."
                "blinking" -> "초록불이 깜빡입니다. 서두르세요."
                null -> {
                    // 직전 신호가 빨간불이었거나 차량이 최근 감지된 경우 → 안전 우선
                    val vehicleRecent  = SystemClock.elapsedRealtime() - lastVehicleDetectMs < 3_000L
                    val wasRed         = lastKnownSignalColor == "red"
                    if (vehicleRecent || wasRed) "신호등을 확인할 수 없습니다. 주변을 살펴보세요."
                    else "신호등을 확인할 수 없습니다. 건너셔도 됩니다."
                }
                else       -> return
            }
        } else {
            // 횡단보도 밖: null(없음)은 무시
            when (color) {
                "red"      -> "빨간불입니다. 멈추세요."
                "green"    -> "초록불입니다. 건너도 됩니다."
                "blinking" -> "초록불이 깜빡입니다. 서두르세요."
                else       -> return
            }
        }
        tts.speak(msg, TextToSpeech.QUEUE_ADD, null, "tl-${color ?: "none"}")
    }

    private fun handleSegmentation(seg: com.safestep.app.detect.SegmentResult) {
        runOnUiThread { segmentOverlay.setImageBitmap(seg.maskBitmap) }
        if (seg.zones.isNotEmpty()) lastZones = seg.zones

        val nowMs = SystemClock.elapsedRealtime()

        // ① 계단 — 최우선 FLUSH
        if (seg.isStairs) {
            if (lastSurfaceStatus != "stairs") {
                lastSurfaceStatus = "stairs"
                onCrosswalk = false   // 계단 진입 시 횡단보도 상태 해제 → 신호등 오발화 방지
                tts.speak("계단입니다. 주의하세요.", TextToSpeech.QUEUE_FLUSH, null, "stairs")
            }
            return
        }

        val status = seg.status

        // ② 위험 노면(차도/자전거도로) 머무는 동안 5초마다 방향 유도 반복
        val isDangerous = status == "road" || status == "bicycle_road"
        if (isDangerous && status == lastSurfaceStatus) {
            if (nowMs - lastSurfaceRepeatMs >= SURFACE_REPEAT_MS) {
                lastSurfaceRepeatMs = nowMs
                tts.speak(buildDirectionGuidance(status), TextToSpeech.QUEUE_ADD, null, "repeat-$nowMs")
            }
            return
        }

        // ③ 노면 변화 없으면 종료
        if (status == lastSurfaceStatus) return
        val prevStatus    = lastSurfaceStatus
        lastSurfaceStatus = status
        lastSurfaceRepeatMs = nowMs

        // ④ 진입 안내
        when (status) {
            "crosswalk" -> {
                onCrosswalk = true
                tts.speak("횡단보도입니다. 멈추고 신호를 확인하세요.",
                    TextToSpeech.QUEUE_FLUSH, null, "crosswalk")
            }
            "road" -> {
                onCrosswalk = false
                tts.speak("차도입니다. 주의하세요.", TextToSpeech.QUEUE_ADD, null, "road-entry")
            }
            "bicycle_road" -> {
                onCrosswalk = false
                tts.speak("자전거 도로입니다. 주의하세요.", TextToSpeech.QUEUE_ADD, null, "bike-entry")
            }
            "alley" -> {
                onCrosswalk = false
                tts.speak("골목길입니다. 주의하세요.", TextToSpeech.QUEUE_ADD, null, "alley")
            }
            "sidewalk" -> {
                onCrosswalk = false
                // 위험 구역에서 복귀할 때만 "인도입니다" 발화 (항상 말하면 너무 잦음)
                if (prevStatus == "road" || prevStatus == "bicycle_road" ||
                    prevStatus == "crosswalk" || prevStatus == "caution") {
                    tts.speak("인도입니다.", TextToSpeech.QUEUE_ADD, null, "sidewalk")
                }
            }
            "caution" -> {
                onCrosswalk = false
                tts.speak("주의구역입니다. 천천히 이동하세요.", TextToSpeech.QUEUE_ADD, null, "caution")
            }
        }
    }

    /**
     * 차도/자전거도로 머무는 동안 반복 발화할 방향 유도 문자열.
     * 좌우 zones 노면 정보를 기반으로 동적 생성.
     */
    private fun buildDirectionGuidance(status: String): String {
        val prefix = if (status == "bicycle_road") "자전거 도로입니다." else "차도입니다."
        val left  = lastZones["left"]  ?: "unknown"
        val right = lastZones["right"] ?: "unknown"

        val lSafe    = left  == "sidewalk" || left  == "alley"
        val rSafe    = right == "sidewalk" || right == "alley"
        val lCross   = left  == "crosswalk"
        val rCross   = right == "crosswalk"
        val lCaution = left  == "caution"
        val rCaution = right == "caution"

        return when {
            lSafe && rSafe    -> "$prefix 왼쪽 또는 오른쪽 인도로 이동하세요."
            lSafe             -> "$prefix 왼쪽 인도로 이동하세요."
            rSafe             -> "$prefix 오른쪽 인도로 이동하세요."
            lCross && rCross  -> "$prefix 횡단보도 방향으로 이동하세요."
            lCross            -> "$prefix 왼쪽 횡단보도 방향으로 이동하세요."
            rCross            -> "$prefix 오른쪽 횡단보도 방향으로 이동하세요."
            lCaution && !rCaution -> "$prefix 왼쪽 골목 방향으로 이동하세요."
            rCaution && !lCaution -> "$prefix 오른쪽 골목 방향으로 이동하세요."
            else              -> "$prefix 천천히 멈추고 주변을 확인하세요."
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 3구역 회피 방향 판단
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 클라이언트 면적 증가율 (서버 depth 없을 때 접근 속도 fallback)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /**
     * 동일 라벨 bbox 면적의 시간당 변화율 (화면 비율/초).
     * 양수 = 커지는 중 = 접근, 충분한 샘플(≥3)이 쌓일 때까지 null.
     */
    private fun areaGrowthRate(label: String, area: Float): Float? {
        val hist = areaHistory.getOrPut(label) { ArrayDeque() }
        val now  = SystemClock.elapsedRealtime()
        hist.addLast(area to now)
        while (hist.size > 8) hist.removeFirst()
        // 3초보다 오래된 샘플 제거
        while (hist.size > 1 && now - hist.first().second > 3_000L) hist.removeFirst()
        if (hist.size < 3) return null
        val dt = (now - hist.first().second) / 1000f
        if (dt < 0.15f) return null
        return (area - hist.first().first) / dt
    }

    /** 회피 방향 TTS를 짧은 형태로 변환 ("오른쪽으로" 등) */
    private fun dodgeHintShort(hint: String?): String? = when {
        hint == null                      -> null
        hint.contains("오른쪽으로 조심히") -> "오른쪽 조심"
        hint.contains("왼쪽으로 조심히")  -> "왼쪽 조심"
        hint.contains("오른쪽")          -> "오른쪽으로"
        hint.contains("왼쪽")            -> "왼쪽으로"
        hint.contains("멈추세요")         -> "멈춰"
        else                             -> null
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 내비게이션 유틸
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /** 위경도 간 평면 근사 거리 (m) — 수백 m 이내 정확 */
    private fun distLatLon(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val dLat = (lat2 - lat1) * 111320.0
        val dLon = (lon2 - lon1) * 111320.0 * Math.cos(Math.toRadians(lat1))
        return Math.sqrt(dLat * dLat + dLon * dLon)
    }

    /**
     * 사용자와 가장 가까운 경로 포인트 이후 구간만 폴리라인에 남김.
     * GPS 이동에 따라 지나온 선이 사라지는 효과.
     */
    private fun updateRemainingRoute(lat: Double, lon: Double) {
        val pts = allPathPoints
        if (pts.size < 2) return

        // 가장 가까운 포인트 인덱스 탐색
        var nearestIdx = 0
        var minDist = Double.MAX_VALUE
        for (i in pts.indices) {
            val d = distLatLon(lat, lon, pts[i].latitude, pts[i].longitude)
            if (d < minDist) { minDist = d; nearestIdx = i }
        }

        val remaining = pts.subList(nearestIdx, pts.size)
        runOnUiThread {
            routePolyline?.setPoints(remaining)
            mapView.invalidate()
        }
    }

    /**
     * 경로 이탈 감지 — 경로상 모든 포인트와의 최소 거리가 40m 초과 시 TTS 경고 + 자동 재탐색.
     * 20초 쿨다운 적용.
     */
    private fun checkOffRoute(lat: Double, lon: Double) {
        val pts = allPathPoints
        if (pts.isEmpty()) return
        val now = SystemClock.elapsedRealtime()
        if (now - lastOffRouteWarnMs < 20_000L) return

        val minDist = pts.minOf { p -> distLatLon(lat, lon, p.latitude, p.longitude) }
        if (minDist > 40.0) {
            lastOffRouteWarnMs = now
            tts.speak("경로를 벗어났습니다. 새로운 경로를 탐색합니다.", TextToSpeech.QUEUE_FLUSH, null, "off-route")
            val dest = currentDest ?: return
            fetchRoute(lat, lon, dest)
        }
    }

    /**
     * 반대 방향 감지.
     * 이전 GPS 위치와 현재 위치로 이동 방위각을 계산하고,
     * 다음 경로 지점 방위각과 150도 이상 차이나면 "반대 방향" 경고.
     *
     * 오감지 방지 조건:
     *  - 스텝 바뀐 직후 10초는 무시
     *  - 이전 위치에서 3m 이상 이동한 경우만 계산
     *  - 15초 쿨다운
     */
    private fun checkWrongWay(loc: Location) {
        if (!navGuide.isRunning()) { prevNavLocation = null; return }
        val now = SystemClock.elapsedRealtime()
        if (now - lastStepChangeMs   < 10_000L) return   // 방금 스텝 바뀜 — 아직 회전 중
        if (now - lastWrongWayWarnMs < 15_000L) return   // 쿨다운

        val prev = prevNavLocation
        prevNavLocation = loc

        if (prev == null) return
        val moved = prev.distanceTo(loc)
        if (moved < 3f) return   // 너무 조금 이동 — GPS 노이즈 가능성

        val step = navGuide.currentStep() ?: return
        val userBearing  = bearingTo(prev.latitude, prev.longitude, loc.latitude,  loc.longitude)
        val routeBearing = bearingTo(loc.latitude,  loc.longitude,  step.lat,      step.lon)
        val diff = angleDiff(userBearing, routeBearing)

        if (diff > 150f) {
            lastWrongWayWarnMs = now
            tts.speak("반대 방향입니다. 돌아서세요.", TextToSpeech.QUEUE_FLUSH, null, "wrong-way-$now")
        }
    }

    /** 두 좌표 간 방위각 (0=북, 시계방향, 0~360) */
    private fun bearingTo(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Float {
        val dLon = Math.toRadians(lon2 - lon1)
        val lat1r = Math.toRadians(lat1)
        val lat2r = Math.toRadians(lat2)
        val x = Math.sin(dLon) * Math.cos(lat2r)
        val y = Math.cos(lat1r) * Math.sin(lat2r) - Math.sin(lat1r) * Math.cos(lat2r) * Math.cos(dLon)
        return ((Math.toDegrees(Math.atan2(x, y)) + 360) % 360).toFloat()
    }

    /** 두 방위각의 최소 각도 차이 (0~180) */
    private fun angleDiff(a: Float, b: Float): Float {
        var d = Math.abs(a - b) % 360f
        if (d > 180f) d = 360f - d
        return d
    }

    /** 노면 카테고리의 위험도 점수 (높을수록 위험) */
    private fun zoneDanger(cat: String): Int = when (cat) {
        "caution"   -> 3
        "road"      -> 2
        "alley"     -> 1
        "crosswalk" -> 1
        "sidewalk"  -> 0
        else        -> 0   // unknown
    }

    /**
     * 장애물이 있는 방향을 받아 반대쪽 구역 안전도를 확인하고 회피 TTS 문자열 반환.
     * 구역 정보가 없으면 null (발화 안 함).
     */
    private fun buildDodgeHint(obstacleSide: String): String? {
        val zones = lastZones
        if (zones.isEmpty()) return null

        val leftDanger  = zoneDanger(zones["left"]   ?: "unknown")
        val rightDanger = zoneDanger(zones["right"]  ?: "unknown")

        return when (obstacleSide) {
            "왼쪽" -> when {
                rightDanger == 0 -> "오른쪽으로 피하세요."
                rightDanger == 1 -> "오른쪽으로 조심히 피하세요."
                else             -> "멈추세요."
            }
            "오른쪽" -> when {
                leftDanger == 0  -> "왼쪽으로 피하세요."
                leftDanger == 1  -> "왼쪽으로 조심히 피하세요."
                else             -> "멈추세요."
            }
            else -> when {   // 정면
                leftDanger < rightDanger  -> "왼쪽으로 피하세요."
                rightDanger < leftDanger  -> "오른쪽으로 피하세요."
                leftDanger == 0           -> "왼쪽으로 피하세요."
                else                      -> "멈추세요."
            }
        }
    }


    private fun vibrateForDirection(side: String, amplitude: Int = 180) {
        val timings = when (side) {
            "왼쪽"  -> longArrayOf(0, 120, 80, 120, 80, 350)
            "오른쪽" -> longArrayOf(0, 350, 80, 120, 80, 120)
            else    -> longArrayOf(0, 250, 80, 100, 80, 250)
        }
        // amplitude 배열: 대기(0) → 진동(amp) → 대기(0) → ... 교번
        val amplitudes = IntArray(timings.size) { i -> if (i % 2 == 1) amplitude else 0 }
        vibrator?.vibrate(VibrationEffect.createWaveform(timings, amplitudes, -1))
    }

    // ══════════════════════════════════════════════════════════════════════════
    // TTS
    // ══════════════════════════════════════════════════════════════════════════

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.KOREAN
            tts.setSpeechRate(1.5f)
            tts.speak(
                "SafeStep 시작됩니다. 마이크 버튼을 눌러 목적지를 말씀해주세요.",
                TextToSpeech.QUEUE_FLUSH, null, "intro"
            )
            // 앱 시작 후 1.5초 뒤 서버 연결 확인
            Handler(Looper.getMainLooper()).postDelayed({ checkServerHealth() }, 1_500)
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 서버 연결 상태 모니터링
    // ══════════════════════════════════════════════════════════════════════════

    private fun checkServerHealth() {
        lastServerCheckMs = SystemClock.elapsedRealtime()
        val healthUrl = RemoteDetector.SERVER_URL
            .substringBeforeLast("/detect")
            .substringBeforeLast("/segment") + "/health"

        Thread {
            val wasConnected = serverConnected
            var ok = false
            try {
                val client = okhttp3.OkHttpClient.Builder()
                    .connectTimeout(3, java.util.concurrent.TimeUnit.SECONDS)
                    .readTimeout(3, java.util.concurrent.TimeUnit.SECONDS)
                    .build()
                val req  = okhttp3.Request.Builder().url(healthUrl).get().build()
                val resp = client.newCall(req).execute()
                ok = resp.isSuccessful
                resp.close()
            } catch (e: Exception) {
                Log.w(TAG, "서버 상태 확인 실패: ${e.message}")
            }

            serverConnected = ok
            runOnUiThread {
                // Activity가 이미 소멸된 경우 TTS 접근 금지 (shutdown 이후 speak 호출 방지)
                if (isDestroyed) return@runOnUiThread
                when {
                    // ① 최초 연결 성공
                    !wasConnected && ok ->
                        tts.speak(
                            "서버에 연결되었습니다. 모든 기능을 사용할 수 있습니다.",
                            TextToSpeech.QUEUE_ADD, null, "srv-ok"
                        )
                    // ② 재연결 성공 (wasConnected가 false였던 경우와 구분할 수 없어 같은 메시지)
                    // ③ 연결 끊김 (이전엔 됐는데 지금 안 됨)
                    wasConnected && !ok ->
                        tts.speak(
                            "서버 연결이 끊겼습니다. 장애물 탐지, 노면 안내, 신호등 인식 기능을 사용할 수 없습니다. 주의하세요.",
                            TextToSpeech.QUEUE_ADD, null, "srv-fail"
                        )
                    // ④ 앱 시작부터 연결 안 됨 (wasConnected=false, ok=false)
                    !wasConnected && !ok ->
                        tts.speak(
                            "서버에 연결할 수 없습니다. 장애물 탐지, 노면 안내, 신호등 인식 기능을 사용할 수 없습니다.",
                            TextToSpeech.QUEUE_ADD, null, "srv-init-fail"
                        )
                }
            }
        }.start()
    }

    /** 서버가 끊긴 상태에서 30초마다 반복 경고 — analyzeFrame() 주기 체크에서 호출 */
    private fun repeatServerOfflineWarning() {
        if (!serverConnected) {
            tts.speak("서버에 연결할 수 없습니다. 주의하세요.",
                TextToSpeech.QUEUE_ADD, null, "srv-repeat-${SystemClock.elapsedRealtime()}")
        }
    }
}
