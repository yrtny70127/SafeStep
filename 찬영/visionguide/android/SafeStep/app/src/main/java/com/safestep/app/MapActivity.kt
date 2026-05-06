package com.safestep.app

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.hardware.camera2.CameraManager
import android.location.LocationManager
import android.os.Handler
import android.os.Looper
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.os.BatteryManager
import android.os.Bundle
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
    private lateinit var destinationText: TextView
    private lateinit var destMicButton: Button
    private lateinit var warningBanner: LinearLayout
    private lateinit var warningText: TextView
    private lateinit var navArrow: TextView
    private lateinit var navInstruction: TextView
    private lateinit var navDistance: TextView
    private lateinit var navStepCount: TextView
    private lateinit var screenOffButton: Button
    private lateinit var screenOffOverlay: View

    // ── Core ──────────────────────────────────────────────────────────────────
    private lateinit var tts: TextToSpeech
    private lateinit var cameraExecutor: ExecutorService
    private var vibrator: Vibrator? = null
    private lateinit var detector: ObjectDetector
    private var remoteDetector: com.safestep.app.detect.RemoteDetector? = null  // 캐스팅 캐시
    private lateinit var segClient: SegmentationClient

    // ── Location ──────────────────────────────────────────────────────────────
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var locationCallback: LocationCallback
    private var currentLocation: Location? = null

    // ── Compass (Heading-up) ──────────────────────────────────────────────────
    private lateinit var sensorManager: SensorManager
    private var rotationSensor: Sensor? = null
    @Volatile private var currentAzimuth = 0f
    private val rotMatrix  = FloatArray(9)
    private val orientVals = FloatArray(3)

    // ── Map overlays ──────────────────────────────────────────────────────────
    private var myLocationOverlay: MyLocationNewOverlay? = null
    private var routePolyline: Polyline? = null
    private var destMarker: Marker? = null

    // ── Navigation ────────────────────────────────────────────────────────────
    private lateinit var navGuide: NavigationGuide
    private var totalSteps = 0
    private var currentDest: PoiResult? = null

    // ── Speech ────────────────────────────────────────────────────────────────
    private var speechRecognizer: SpeechRecognizer? = null
    private var isListening = false

    // ── 대중교통 안내 응답 대기 상태 ─────────────────────────────────────────
    @Volatile private var awaitingTransitConfirm = false
    private var pendingRoute: RouteResult? = null
    private var pendingDest: PoiResult? = null
    private val transitConfirmHandler = Handler(Looper.getMainLooper())
    // 현재 안내 중인 목적지가 정류장(임시 경유)인지 → 도착 시 별도 음성
    @Volatile private var navigatingToTransit = false
    private var transitStopName: String = ""

    // ── Segmentation ──────────────────────────────────────────────────────────
    @Volatile private var frameCount = 0
    @Volatile private var lastSurfaceStatus = ""   // frontCls 또는 status (세밀한 변화 감지)

    // ── 거리 기반 음성 쿨다운 ─────────────────────────────────────────────────
    @Volatile private var alert5mFired = false
    @Volatile private var alert1mFired = false
    @Volatile private var lastTrackedLabel = ""
    // 1m 긴급: 같은 라벨에 대해 3초 쿨다운 (명세 1순위)
    @Volatile private var lastUrgentLabel  = ""
    @Volatile private var lastUrgentSpokeMs = 0L
    private val URGENT_COOLDOWN_MS = 3_000L
    // 5m 진입: 같은 라벨이 한번 5m 밖으로 나간 뒤 다시 들어와야 재발화
    private val labels5mFired = mutableSetOf<String>()

    // ── 서버 연결 상태 ────────────────────────────────────────────────────────
    @Volatile private var wasServerConnected = true
    @Volatile private var serverStateAnnounced = false
    @Volatile private var wasServerBusy = false

    // ── GPS 상태 ──────────────────────────────────────────────────────────────
    private var wasGpsEnabled = true
    private lateinit var locationManager: LocationManager
    @Volatile private var lowAccuracySinceMs = 0L
    @Volatile private var indoorAnnounced = false

    // ── Depth 상태 ────────────────────────────────────────────────────────────
    @Volatile private var depthNullStreak = 0
    @Volatile private var depthUnavailSpoken = false
    @Volatile private var lastDepthAutoOff = false  // 서버 자동 OFF 추적

    // ── 신호등 색상 (횡단보도) ────────────────────────────────────────────────
    // "unset": 횡단보도 미진입 or 초기값 / "": 신호 인식 불가 / "red"/"green": 신호
    @Volatile private var lastTrafficLightColor = "unset"

    // ── 차량 근접 여부 (횡단보도 "건너셔도 됩니다" 판단용) ──────────────────
    @Volatile private var hasNearbyVehicle = false

    // ── 서버 재연결 실패 30초 반복 ────────────────────────────────────────────
    private val serverReconnectHandler = Handler(Looper.getMainLooper())

    // ── 진동 / 차도 반복 컨트롤러 (GuideHelper.kt) ──────────────────────────
    private lateinit var vibCtrl:  VibrationController
    private lateinit var roadCtrl: RoadRepeatController

    // ── 배터리 절감: 적응형 해상도 + FPS ─────────────────────────────────
    @Volatile private var hiResRemaining   = 0       // 고해상(640) 남은 프레임
    @Volatile private var lastFrameSentMs  = 0L
    @Volatile private var lastThumb: IntArray? = null
    private val LOW_RES_MAX_EDGE  = 320
    private val HI_RES_MAX_EDGE   = 640
    private val HI_RES_FRAMES_AFTER_DANGER = 10
    private val SCENE_DIFF_THRESHOLD       = 8       // 평균 명도 차이 (0~255)

    // ── 배터리 절감: 화면 끄기 모드 (5️⃣-A) ──────────────────────────────
    @Volatile private var screenOffMode = false

    // ── 카메라 밝기 / 손전등 ──────────────────────────────────────────────
    private var cameraManager: CameraManager? = null
    private var torchCameraId: String? = null
    @Volatile private var torchOn = false
    @Volatile private var darkSinceMs = 0L         // 어두움 지속 시작 시각
    @Volatile private var torchDarkSinceMs = 0L    // 손전등 ON 후 어두움 지속 시각
    @Volatile private var emptyResultStreak = 0    // 밝기 정상인데 탐지·세그 둘 다 빈 결과 프레임 수
    @Volatile private var blockSpoken = false      // 가림 안내 발화 여부
    @Volatile private var lastBlockSpokeMs = 0L

    // ── 배터리 / 절전 모드 ────────────────────────────────────────────────
    @Volatile private var lastBatteryAlertLevel = 100   // 마지막으로 알린 임계 레벨
    @Volatile private var powerSaveMode = false
    @Volatile private var lastBatteryCriticalSpokeMs = 0L  // 5% 5초 반복용
    @Volatile private var isCharging = false

    companion object {
        private const val TAG      = "MapActivity"
        private const val REQ_PERM = 200
        // 경고 그룹 분류 (Detection.label = 한국어 그룹명)
        private val FULL_ALERT_GROUPS   = setOf("차량", "개인이동장치") // 5m 음성+진동 + 1m 긴급
        private val PERSON_ALERT_GROUPS = setOf("사람/동물")            // 1m 긴급 음성만
        private val CLOSE_ALERT_GROUPS  = setOf("고정장애물")           // 1m 긴급 음성만
        private val SIGNAL_ALERT_GROUPS = setOf("신호등/표지판")        // 1m 진동만
        // 기타 → 알림 없음
        // 위험도 점수 그룹 가중치
        private val GROUP_WEIGHT = mapOf(
            "차량"          to 1.0f,
            "개인이동장치"   to 0.9f,
            "사람/동물"      to 0.7f,
            "고정장애물"      to 0.5f,
            "신호등/표지판"  to 0.3f,
        )
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
        destinationText = findViewById(R.id.destinationText)
        destMicButton   = findViewById(R.id.destMicButton)
        warningBanner   = findViewById(R.id.warningBanner)
        warningText     = findViewById(R.id.warningText)
        navArrow        = findViewById(R.id.navArrow)
        navInstruction  = findViewById(R.id.navInstruction)
        navDistance     = findViewById(R.id.navDistance)
        navStepCount    = findViewById(R.id.navStepCount)
        screenOffButton  = findViewById(R.id.screenOffButton)
        screenOffOverlay = findViewById(R.id.screenOffOverlay)

        findViewById<Button>(R.id.backButton).setOnClickListener {
            startActivity(Intent(this, SplashActivity::class.java))
            finish()
        }

        // 화면 끄기 토글
        screenOffButton.setOnClickListener { toggleScreenOff(true) }
        // 검정 오버레이 어디든 탭 → 해제
        screenOffOverlay.setOnClickListener { toggleScreenOff(false) }

        tts            = TextToSpeech(this, this)
        @Suppress("DEPRECATION")
        vibrator       = getSystemService(VIBRATOR_SERVICE) as Vibrator
        cameraExecutor = Executors.newSingleThreadExecutor()
        detector       = ObjectDetector.create(this)
        remoteDetector = detector as? com.safestep.app.detect.RemoteDetector
        segClient      = SegmentationClient(RemoteDetector.SERVER_URL)
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        // 공용 컨트롤러 (GuideHelper.kt) — vibrator는 위에서 초기화됨, tts는 onInit 이전에도 객체는 존재
        vibCtrl  = VibrationController(vibrator)
        roadCtrl = RoadRepeatController(tts, Handler(Looper.getMainLooper()))

        // 나침반 센서
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        locationManager = getSystemService(LOCATION_SERVICE) as LocationManager

        // ── 카메라 매니저 (손전등 제어용) — 후면 카메라 ID 찾기 ──
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
        torchCameraId = runCatching {
            cameraManager?.cameraIdList?.firstOrNull { id ->
                val ch = cameraManager!!.getCameraCharacteristics(id)
                ch.get(android.hardware.camera2.CameraCharacteristics.LENS_FACING) ==
                    android.hardware.camera2.CameraCharacteristics.LENS_FACING_BACK &&
                    (ch.get(android.hardware.camera2.CameraCharacteristics.FLASH_INFO_AVAILABLE) ?: false)
            }
        }.getOrNull()

        navGuide = NavigationGuide(tts).apply {
            onStepChanged = { step ->
                runOnUiThread { updateNavUI(step.description, step.distance) }
            }
            onArrived = { runOnUiThread { onArrived() } }
            onOffRoute = reroute@{
                val dest = currentDest ?: return@reroute
                val loc  = currentLocation ?: return@reroute
                tts.speak("경로를 이탈했습니다. 재탐색합니다.", TextToSpeech.QUEUE_FLUSH, null, "reroute")
                fetchRoute(loc.latitude, loc.longitude, dest)
            }
            onWrongDirection = {
                tts.speak("반대 방향입니다. 돌아서세요.", TextToSpeech.QUEUE_FLUSH, null, "wrong-dir")
            }
        }

        setupMap()
        setupMicButton()
        buildLocationCallback()

        // 화면 자동 꺼짐 방지 (절전 모드에서 자동 해제)
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        if (hasAllPermissions()) startAll()
        else ActivityCompat.requestPermissions(this, PERMISSIONS, REQ_PERM)
    }

    override fun onResume() {
        super.onResume()
        mapView.onResume()
        rotationSensor?.let {
            sensorManager.registerListener(compassListener, it, SensorManager.SENSOR_DELAY_UI)
        }
        registerReceiver(gpsStateReceiver, IntentFilter(LocationManager.PROVIDERS_CHANGED_ACTION))
        registerReceiver(batteryReceiver, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
    }

    override fun onPause() {
        super.onPause()
        mapView.onPause()
        sensorManager.unregisterListener(compassListener)
        runCatching { unregisterReceiver(gpsStateReceiver) }
        runCatching { unregisterReceiver(batteryReceiver) }
        // 백그라운드 진입 시 손전등 강제 OFF (사용자에게 음성 안내 없음)
        setTorch(false)
    }

    override fun onDestroy() {
        roadCtrl.stop()
        stopServerReconnectAlert()
        transitConfirmHandler.removeCallbacksAndMessages(null)
        setTorch(false)                    // 종료 시 손전등 강제 OFF
        mapView.onDetach()
        cameraExecutor.shutdown()
        tts.shutdown()
        speechRecognizer?.destroy()
        if (::fusedLocationClient.isInitialized && ::locationCallback.isInitialized)
            fusedLocationClient.removeLocationUpdates(locationCallback)
        runCatching { detector.close() }
        runCatching { segClient.close() }   // OkHttp 풀 정리
        super.onDestroy()
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 나침반 → Heading-up 지도 회전
    // ══════════════════════════════════════════════════════════════════════════

    private val compassListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            SensorManager.getRotationMatrixFromVector(rotMatrix, event.values)
            SensorManager.getOrientation(rotMatrix, orientVals)
            val az = Math.toDegrees(orientVals[0].toDouble()).toFloat()
            currentAzimuth = (az + 360f) % 360f
            mapView.setMapOrientation(-currentAzimuth)
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    // ── GPS 상태 변화 감지 ────────────────────────────────────────────────────
    private val gpsStateReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action != LocationManager.PROVIDERS_CHANGED_ACTION) return
            val enabled = locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)
            if (enabled && !wasGpsEnabled) {
                wasGpsEnabled = true
                if (navGuide.isRunning()) {
                    // 내비 중이었으면 쿨다운 초기화 → 다음 위치 업데이트에서 즉시 이탈 감지
                    navGuide.resetOffRouteCooldown()
                    tts.speak("위치 정보가 연결되었습니다. 경로를 확인합니다.",
                        TextToSpeech.QUEUE_ADD, null, "gps-on")
                } else {
                    tts.speak("위치 정보가 연결되었습니다. 목적지 안내를 사용할 수 있습니다.",
                        TextToSpeech.QUEUE_ADD, null, "gps-on")
                }
            } else if (!enabled && wasGpsEnabled) {
                wasGpsEnabled = false
                tts.speak("위치 정보 연결이 끊겼습니다. 목적지 안내를 사용할 수 없습니다. 주의하세요.",
                    TextToSpeech.QUEUE_FLUSH, null, "gps-off")
            }
        }
    }

    // ── 앱 시작 초기 상태 안내 ────────────────────────────────────────────────
    private fun checkInitialStates() {
        wasGpsEnabled = locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)
        if (!wasGpsEnabled) {
            tts.speak("위치 정보를 사용할 수 없습니다. 목적지 안내 기능을 사용할 수 없습니다.",
                TextToSpeech.QUEUE_ADD, null, "gps-init-off")
        }
        // 서버 연결 상태는 첫 프레임 탐지 후 안내 (analyzeFrame에서 처리)
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

    // 진행 방향으로 지도 중심 오프셋 (사용자가 화면 하단 1/3에 보이도록)
    private fun centerMapOnUser(loc: Location) {
        val zoom = mapView.zoomLevelDouble
        // 줌 레벨별 픽셀당 미터 (위도 보정 포함)
        val metersPerPx = 156543.03392 *
                Math.cos(Math.toRadians(loc.latitude)) / Math.pow(2.0, zoom)
        // 화면 높이의 1/3만큼 진행 방향으로 오프셋
        val shiftM = (mapView.height / 3.0) * metersPerPx
        val bearingRad = Math.toRadians(currentAzimuth.toDouble())
        val dLat = shiftM / 111320.0 * Math.cos(bearingRad)
        val dLon = shiftM / (111320.0 * Math.cos(Math.toRadians(loc.latitude))) *
                Math.sin(bearingRad)
        mapView.controller.animateTo(GeoPoint(loc.latitude + dLat, loc.longitude + dLon))
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Location
    // ══════════════════════════════════════════════════════════════════════════

    private fun buildLocationCallback() {
        locationCallback = object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                val loc = result.lastLocation ?: return
                currentLocation = loc

                // ── GPS 정확도 추적 (50m 초과 30초 지속 → 실내 안내) ──
                val now = SystemClock.elapsedRealtime()
                if (loc.accuracy > 50f) {
                    if (lowAccuracySinceMs == 0L) lowAccuracySinceMs = now
                    if (!indoorAnnounced && now - lowAccuracySinceMs >= 30_000L) {
                        indoorAnnounced = true
                        tts.speak("실내인 것 같습니다. 출구를 찾으면 다시 안내합니다.",
                            TextToSpeech.QUEUE_ADD, null, "indoor")
                    }
                } else {
                    if (indoorAnnounced) {
                        indoorAnnounced = false
                        tts.speak("위치 정보가 연결되었습니다. 목적지 안내를 사용할 수 있습니다.",
                            TextToSpeech.QUEUE_ADD, null, "outdoor")
                    }
                    lowAccuracySinceMs = 0L
                }

                // 실내(저정확도) 상태에서는 navGuide 업데이트 스킵 (목적지 안내 일시 중단)
                if (!indoorAnnounced) navGuide.updateLocation(loc.latitude, loc.longitude)

                // 지도 중심 이동 (진행 방향 기준 오프셋)
                centerMapOnUser(loc)

                // 내비 중 줌 자동 조정
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
                }
            }
        }
    }

    /**
     * 위치 업데이트 시작/재구성.
     * - 내비 안내 중: HIGH_ACCURACY 1초
     * - 일반 모드(목적지 없을 때): BALANCED_POWER_ACCURACY 5초 (배터리 절감)
     */
    private fun startLocationUpdates(highAccuracy: Boolean = false) {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED) return
        // 기존 콜백 해제 후 재구성
        runCatching { fusedLocationClient.removeLocationUpdates(locationCallback) }
        val req = if (highAccuracy) {
            LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1_000L)
                .setMinUpdateDistanceMeters(1f)
                .build()
        } else {
            LocationRequest.Builder(Priority.PRIORITY_BALANCED_POWER_ACCURACY, 5_000L)
                .setMinUpdateDistanceMeters(5f)
                .build()
        }
        fusedLocationClient.requestLocationUpdates(req, locationCallback, mainLooper)
        Log.i(TAG, "GPS 모드: ${if (highAccuracy) "HIGH (1s)" else "BALANCED (5s)"}")
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
        // 정류장 도착이면 별도 안내 (NavigationGuide의 기본 "목적지에 도착" 외 추가 음성)
        if (navigatingToTransit) {
            val name = transitStopName.ifEmpty { "정류장" }
            tts.speak("$name 에 도착했습니다. 대중교통 이용 후 다시 목적지를 말씀해주세요.",
                TextToSpeech.QUEUE_ADD, null, "transit-arrived")
            navigatingToTransit = false
            transitStopName     = ""
        }

        navArrow.text        = "🏁"
        navInstruction.text  = "목적지 도착"
        navDistance.text     = ""
        navStepCount.text    = ""
        destinationText.text = "목적지를 말씀해주세요"
        mapView.controller.setZoom(18.0)
        // 내비 종료 → GPS 절전 모드로 복귀
        startLocationUpdates(highAccuracy = false)
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
        tts.speak("$query 검색합니다.", TextToSpeech.QUEUE_FLUSH, null, "searching")

        TmapService.searchPoi(query) { pois ->
            if (pois.isEmpty()) {
                tts.speak("검색 결과가 없습니다. 다시 말씀해주세요.", TextToSpeech.QUEUE_FLUSH, null, "no-poi")
                destinationText.text = "목적지를 말씀해주세요"
                return@searchPoi
            }
            val dest = pois.first()
            currentDest = dest
            destinationText.text = dest.name
            tts.speak("${dest.name}(으)로 경로를 탐색합니다.", TextToSpeech.QUEUE_FLUSH, null, "found")

            val loc = currentLocation
            if (loc == null) {
                tts.speak("현재 위치를 가져오는 중입니다. 잠시 후 다시 시도해주세요.",
                    TextToSpeech.QUEUE_ADD, null, "no-loc")
                return@searchPoi
            }
            fetchRoute(loc.latitude, loc.longitude, dest)
        }
    }

    private fun fetchRoute(sLat: Double, sLon: Double, dest: PoiResult) {
        TmapService.searchPedestrianRoute(sLat, sLon, dest.lat, dest.lon, dest.name) { result ->
            if (result == null || result.steps.isEmpty()) {
                tts.speak("경로를 찾을 수 없습니다.", TextToSpeech.QUEUE_FLUSH, null, "no-route")
                return@searchPedestrianRoute
            }
            // ── 도보 2km 또는 30분 초과 → 대중교통 안내 옵션 제시 ──
            val isFar = result.totalDistanceM >= 2_000 || result.totalTimeSec >= 30 * 60
            if (isFar && !awaitingTransitConfirm) {
                pendingRoute = result
                pendingDest  = dest
                awaitingTransitConfirm = true
                val km = "%.1f".format(result.totalDistanceM / 1000.0)
                tts.speak(
                    "목적지까지 ${km}킬로미터입니다. 도보로 30분 이상 걸립니다. " +
                    "가까운 정류장으로 안내할까요? 네 또는 아니오로 답해주세요.",
                    TextToSpeech.QUEUE_FLUSH, null, "transit-prompt")
                startTransitConfirmListening()
                return@searchPedestrianRoute
            }
            startNavigation(result, dest)
        }
    }

    /** 경로를 즉시 시작 (도보 또는 정류장 경유 모두 공통) */
    private fun startNavigation(result: RouteResult, dest: PoiResult) {
        totalSteps = result.steps.size
        showRouteOnMap(result, dest)
        navGuide.start(result.steps)
        // 내비 시작 → GPS 고정밀 모드 (1초 간격)
        startLocationUpdates(highAccuracy = true)

        val first = result.steps.first()
        runOnUiThread {
            updateNavUI(first.description, first.distance)
            navStepCount.text = "1/$totalSteps"
        }

        val dist = if (result.totalDistanceM >= 1000)
            "${"%.1f".format(result.totalDistanceM / 1000.0)}km"
        else "${result.totalDistanceM}m"
        val min = result.totalTimeSec / 60
        tts.speak(
            "경로 탐색 완료. 총 $dist, 약 ${min}분 소요됩니다. ${first.description}",
            TextToSpeech.QUEUE_ADD, null, "route-ready"
        )
    }

    // ── 대중교통 확인 음성 인식 ──────────────────────────────────────────────
    private fun startTransitConfirmListening() {
        // 10초 타임아웃 → 도보로 진행
        transitConfirmHandler.postDelayed({
            if (awaitingTransitConfirm) {
                awaitingTransitConfirm = false
                speechRecognizer?.cancel()
                tts.speak("도보로 안내합니다.", TextToSpeech.QUEUE_ADD, null, "transit-no-timeout")
                pendingRoute?.let { r -> pendingDest?.let { d -> startNavigation(r, d) } }
                pendingRoute = null; pendingDest = null
            }
        }, 10_000L)

        runCatching {
            val sr = speechRecognizer ?: return
            sr.cancel()
            sr.setRecognitionListener(transitConfirmListener)
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
            }
            sr.startListening(intent)
        }
    }

    private val transitConfirmListener = object : RecognitionListener {
        override fun onResults(results: Bundle?) {
            if (!awaitingTransitConfirm) return
            val text = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull() ?: ""
            handleTransitConfirm(text)
        }
        override fun onError(error: Int) {
            // 인식 실패 시 도보로 진행
            if (!awaitingTransitConfirm) return
            handleTransitConfirm("")
        }
        override fun onReadyForSpeech(p: Bundle?) {}
        override fun onBeginningOfSpeech() {}
        override fun onRmsChanged(v: Float) {}
        override fun onBufferReceived(b: ByteArray?) {}
        override fun onEndOfSpeech() {}
        override fun onPartialResults(p: Bundle?) {}
        override fun onEvent(t: Int, p: Bundle?) {}
    }

    private fun handleTransitConfirm(text: String) {
        awaitingTransitConfirm = false
        transitConfirmHandler.removeCallbacksAndMessages(null)
        // 마이크 리스너 원복
        speechRecognizer?.setRecognitionListener(destinationRecognitionListener)

        val yes = text.contains("네") || text.contains("응") || text.contains("그래") || text.contains("좋")
        val loc = currentLocation
        val pendingD = pendingDest
        val pendingR = pendingRoute

        if (!yes || loc == null || pendingD == null || pendingR == null) {
            tts.speak("도보로 안내합니다.", TextToSpeech.QUEUE_ADD, null, "transit-no")
            if (pendingR != null && pendingD != null) startNavigation(pendingR, pendingD)
            pendingRoute = null; pendingDest = null
            return
        }

        // 가까운 정류장 검색 (버스정류장 → 없으면 지하철역). 도착 시 별도 음성 위해 플래그 set.
        TmapService.searchNearestTransit(loc.latitude, loc.longitude, "버스정류장", 500) { stop ->
            if (stop != null) {
                tts.speak("${stop.name}으로 안내합니다.", TextToSpeech.QUEUE_ADD, null, "transit-stop")
                currentDest = stop
                navigatingToTransit = true
                transitStopName     = stop.name
                fetchRoute(loc.latitude, loc.longitude, stop)
            } else {
                TmapService.searchNearestTransit(loc.latitude, loc.longitude, "지하철역", 1000) { sub ->
                    if (sub != null) {
                        tts.speak("${sub.name}으로 안내합니다.", TextToSpeech.QUEUE_ADD, null, "transit-sub")
                        currentDest = sub
                        navigatingToTransit = true
                        transitStopName     = sub.name
                        fetchRoute(loc.latitude, loc.longitude, sub)
                    } else {
                        tts.speak("주변에 정류장을 찾지 못했습니다. 도보로 안내합니다.",
                            TextToSpeech.QUEUE_ADD, null, "transit-none")
                        startNavigation(pendingR, pendingD)
                    }
                    pendingRoute = null; pendingDest = null
                }
                return@searchNearestTransit
            }
            pendingRoute = null; pendingDest = null
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

    @ExperimentalGetImage
    private fun analyzeFrame(imageProxy: ImageProxy) {
        try {
            val bitmap   = imageProxy.toBitmap()
            val rotation = imageProxy.imageInfo.rotationDegrees

            // ── 적응형 FPS: GPS 속도 기반 전송 간격 결정 ────────────────────
            val isHiRes  = hiResRemaining > 0
            val now      = SystemClock.elapsedRealtime()
            val speed    = currentLocation?.speed ?: 0f       // m/s
            val targetIntervalMs = when {
                powerSaveMode && speed < 0.5f -> 2_000L       // 절전+정지: 2초
                powerSaveMode                 -> 1_000L       // 절전: 1초
                speed < 0.5f                  -> 1_000L       // 정지: 1초
                speed < 2.0f                  -> 500L         // 보통: 0.5초
                else                          -> 200L         // 빠른 이동: 0.2초
            }
            // hiRes 모드(위험 직후)에선 무조건 통과
            if (!isHiRes && now - lastFrameSentMs < targetIntervalMs) {
                trackCameraState(avgBrightness(bitmap), false)  // 밝기·가림은 계속 추적
                return
            }

            // ── 장면 변화 감지 (보조): 이전과 거의 같은 화면이면 스킵 ──
            val thumb = thumbnail8(bitmap)
            if (!isHiRes && lastThumb != null && thumbDiff(lastThumb!!, thumb) < SCENE_DIFF_THRESHOLD) {
                trackCameraState(avgBrightness(bitmap), false)
                return
            }
            lastThumb = thumb
            lastFrameSentMs = now

            // ── 한 번만 회전+리사이즈+JPEG 압축 → detect/segment 공유 (캐시 명중) ──
            // 파생 bitmap은 사용 후 즉시 recycle (매 프레임 누수 방지)
            val rd = remoteDetector
            val sharedJpeg: ByteArray? = if (rd != null) {
                val rotated = if (rotation != 0) {
                    val m = android.graphics.Matrix().apply { postRotate(rotation.toFloat()) }
                    android.graphics.Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
                } else bitmap
                val maxEdge = if (isHiRes) HI_RES_MAX_EDGE else LOW_RES_MAX_EDGE
                val scale   = maxEdge.toFloat() / kotlin.math.max(rotated.width, rotated.height)
                val scaled  = if (scale < 1f) {
                    android.graphics.Bitmap.createScaledBitmap(
                        rotated,
                        (rotated.width  * scale).toInt(),
                        (rotated.height * scale).toInt(),
                        true
                    )
                } else rotated
                val out = java.io.ByteArrayOutputStream()
                scaled.compress(android.graphics.Bitmap.CompressFormat.JPEG,
                    com.safestep.app.detect.RemoteDetector.JPEG_QUALITY, out)
                // 파생 비트맵 정리 (원본 `bitmap`은 imageProxy.close()와 함께 GC)
                if (scaled !== rotated) scaled.recycle()
                if (rotated !== bitmap) rotated.recycle()
                out.toByteArray()
            } else null

            // hiRes 카운터 감소 (이번 프레임 사용 후)
            if (hiResRemaining > 0) hiResRemaining--

            val detections    = if (sharedJpeg != null) rd!!.detectJpeg(sharedJpeg)
                                else detector.detect(bitmap, rotation)
            val serverMessage = remoteDetector?.lastMessage ?: ""
            val dodgeDir      = remoteDetector?.lastDodge ?: "정면"

            // ── 서버 연결 상태 체크 ──────────────────────────────────────────
            val nowConnected = remoteDetector?.isConnected ?: true
            val nowBusy      = remoteDetector?.isServerBusy ?: false

            // 만석 → 만석 해소 전환 안내
            if (nowBusy && !wasServerBusy) {
                wasServerBusy = true
                tts.speak("서버가 만석입니다. 다른 사용자가 안내 중입니다. 잠시 후 다시 시도해주세요.",
                    TextToSpeech.QUEUE_ADD, null, "server-busy")
                startServerReconnectAlert()
            } else if (!nowBusy && wasServerBusy && nowConnected) {
                wasServerBusy = false
                stopServerReconnectAlert()
                tts.speak("서버에 연결되었습니다. 모든 기능을 사용할 수 있습니다.",
                    TextToSpeech.QUEUE_ADD, null, "server-busy-cleared")
            }

            if (!serverStateAnnounced) {
                serverStateAnnounced = true
                wasServerConnected = nowConnected
                if (nowBusy) {
                    // 첫 응답이 만석 — 위에서 이미 안내됨
                } else if (nowConnected) {
                    tts.speak("서버에 연결되었습니다. 모든 기능을 사용할 수 있습니다.",
                        TextToSpeech.QUEUE_ADD, null, "server-init-ok")
                } else {
                    tts.speak("서버에 연결할 수 없습니다. 장애물 탐지, 노면 안내, 신호등 인식 기능을 사용할 수 없습니다.",
                        TextToSpeech.QUEUE_ADD, null, "server-init-fail")
                }
            } else if (!nowConnected && wasServerConnected) {
                wasServerConnected = false
                tts.speak("서버 연결이 끊겼습니다. 장애물 탐지, 노면 안내, 신호등 인식 기능을 사용할 수 없습니다. 주의하세요.",
                    TextToSpeech.QUEUE_FLUSH, null, "server-off")
                startServerReconnectAlert()
            } else if (nowConnected && !wasServerConnected && !nowBusy) {
                wasServerConnected = true
                stopServerReconnectAlert()
                tts.speak("서버에 다시 연결되었습니다. 모든 기능을 사용할 수 있습니다.",
                    TextToSpeech.QUEUE_ADD, null, "server-on")
            }

            // ── Depth 서버 자동 OFF/ON 안내 ──────────────────────────────
            val nowDepthOff = remoteDetector?.depthAutoOff ?: false
            if (nowDepthOff != lastDepthAutoOff) {
                lastDepthAutoOff = nowDepthOff
                if (nowDepthOff) {
                    tts.speak("서버가 바빠 거리 측정을 일시 중단합니다.",
                        TextToSpeech.QUEUE_ADD, null, "depth-auto-off")
                } else {
                    tts.speak("거리 측정이 다시 활성화되었습니다.",
                        TextToSpeech.QUEUE_ADD, null, "depth-auto-on")
                }
            }

            handleDetections(detections, serverMessage, dodgeDir)

            frameCount++
            // 세그멘테이션: 정상 2프레임마다, 절전 모드는 비활성
            var seg: com.safestep.app.detect.SegmentResult? = null
            if (!powerSaveMode && frameCount % 2 == 0) {
                seg = if (sharedJpeg != null) segClient.segmentJpeg(sharedJpeg)
                      else segClient.segment(bitmap, rotation)
                if (seg != null) handleSegmentation(seg)
            }

            // ── 카메라 밝기 / 가림 추적 ──
            val brightness = avgBrightness(bitmap)
            val anyResult  = detections.isNotEmpty() ||
                             (seg != null && (seg.frontCls.isNotEmpty() || seg.status != "unknown"))
            trackCameraState(brightness, anyResult)
        } catch (e: Exception) {
            Log.e(TAG, "프레임 분석 실패", e)
        } finally {
            imageProxy.close()
        }
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

        // 경고 대상: 기타 그룹만 제외
        val alertCandidates = detections.filter {
            it.label in FULL_ALERT_GROUPS || it.label in PERSON_ALERT_GROUPS ||
            it.label in CLOSE_ALERT_GROUPS || it.label in SIGNAL_ALERT_GROUPS
        }

        // 차량/개인이동장치 근접 여부 (횡단보도 "건너셔도 됩니다" 판단용)
        hasNearbyVehicle = alertCandidates.any { it.label in FULL_ALERT_GROUPS }

        // 5m 이내(또는 depth 모를 때 모두) 후보만 추림
        val nearby = alertCandidates.filter { it.depthM == null || it.depthM <= 5f }
        if (nearby.isEmpty()) {
            alert5mFired = false; alert1mFired = false; lastTrackedLabel = ""
            labels5mFired.clear()  // 모든 객체가 5m 밖 → 다음 진입 시 재발화 가능
            return
        }
        // 5m 안에 더 이상 없는 라벨은 set에서 제거 (다시 들어올 때 재발화 허용)
        val nearbyLabels = nearby.map { it.label }.toSet()
        labels5mFired.retainAll(nearbyLabels)

        // ── 위험도 점수: 면적 × 중심 가까움 × 신뢰도 × 그룹 가중치 ──────────
        val worst = nearby.maxByOrNull { d ->
            val cw  = 1f - kotlin.math.abs(d.centerX() - 0.5f)
            val gw  = GROUP_WEIGHT[d.label] ?: 0.5f
            d.area() * (0.7f + 0.3f * cw) * d.confidence * gw
        } ?: return

        val depthM = worst.depthM
        val group  = worst.label
        val side   = dodgeDir

        // ── 다객체 분석 ────────────────────────────────────────────────────
        val under1m = nearby.filter { it.depthM != null && it.depthM <= 1f }
        val under5m = nearby.filter { it.depthM != null && it.depthM in 1f..5f }
        val sameGroupCount = nearby.count { it.label == group }
        val multipleSameGroup = sameGroupCount >= 3        // "여러 명/여러 대"
        val multipleUnder1m   = under1m.size >= 2          // 가장 가까운 1개만 음성
        val multipleUnder5m   = under5m.size >= 2          // 통합 안내

        // 새로운 장애물 추적 시작 → 쿨다운 초기화
        if (worst.label != lastTrackedLabel) {
            alert5mFired = false; alert1mFired = false
            lastTrackedLabel = worst.label
        }

        // 진동 세기 결정 (depth_m 기반)
        val amplitude = when {
            depthM == null -> 180
            depthM <= 1f   -> 255
            depthM <= 3f   -> 180
            else           -> 100
        }

        // ── 신호등/표지판: 1m 진동만 ──────────────────────────────────────
        if (group in SIGNAL_ALERT_GROUPS) {
            if (depthM != null && depthM <= 1f) vibCtrl.vibrate(side, amplitude)
            return
        }

        // ── 1m 이하 긴급 (같은 라벨 3초 쿨다운, 명세 1순위) ─────────────────
        if (depthM != null && depthM <= 1f) {
            val nowMs = SystemClock.elapsedRealtime()
            val sameLabel  = group == lastUrgentLabel
            val cooledDown = !sameLabel || (nowMs - lastUrgentSpokeMs >= URGENT_COOLDOWN_MS)

            if (!alert1mFired || cooledDown) {
                alert1mFired      = true
                lastUrgentLabel   = group
                lastUrgentSpokeMs = nowMs
                // 위험 직후 N프레임은 고해상도(640)로 정밀 추적
                hiResRemaining = HI_RES_FRAMES_AFTER_DANGER
                val noun = if (multipleSameGroup) "여러 ${groupNoun(group)}" else groupNoun(group)
                val urgentMsg = when {
                    serverMessage.isNotEmpty() && !multipleSameGroup -> serverMessage
                    else -> "위험! $side $noun"  // 짧게 (≤ 2초)
                }
                showWarning(urgentMsg, side, amplitude, speakNow = true)
                if (multipleUnder1m) vibCtrl.vibrate(side, 255)
                return
            }
            // 쿨다운 중이거나 이미 발화 → 진동만
            vibCtrl.vibrate(side, amplitude)
            return
        }

        // ── 사람·고정장애물: 5m 음성 없음 (1m 긴급은 위에서 처리됨) ──────
        if (group in PERSON_ALERT_GROUPS || group in CLOSE_ALERT_GROUPS) return

        // ── 차량/개인이동장치: 5m 진입 음성 (라벨당 1회, 5m 밖 나갔다 들어와야 재발화) ──
        if (group !in labels5mFired) {
            labels5mFired += group
            alert5mFired = true
            // 차량/개인이동장치 5m 진입 → 절반 정도 고해상도로 정밀 추적
            hiResRemaining = HI_RES_FRAMES_AFTER_DANGER / 2
            val msg = when {
                multipleUnder5m ->
                    "여러 장애물 접근. 주의하세요"
                depthUnavailSpoken ->
                    "${side}에 ${groupNoun(group)}"  // 거리 모를 때
                serverMessage.isNotEmpty() -> serverMessage
                else -> {
                    val distStr = if (depthM != null) "${String.format("%.1f", depthM)}m " else ""
                    "$side $distStr${groupNoun(group)} 접근"  // ≤ 1.5초
                }
            }
            showWarning(msg, side, amplitude, speakNow = true)
            return
        }

        // 그 사이: 진동만
        vibCtrl.vibrate(side, amplitude)
    }

    private fun handleSegmentation(seg: com.safestep.app.detect.SegmentResult) {
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
        // 횡단보도 벗어나면 색상 추적 초기화 (다음 진입 시 재발화)
        if (status != "crosswalk") lastTrafficLightColor = "unset"

        // frontCls 우선으로 변화 감지 (세부 클래스 단위 추적)
        val surfaceKey = if (frontCls.isNotEmpty()) frontCls else status
        if (surfaceKey == lastSurfaceStatus) return
        lastSurfaceStatus = surfaceKey

        // 이전 차도/자전거도로 반복 중단
        roadCtrl.stop()

        val message = surfaceMessageFor(frontCls, status)
        if (message != null) {
            // 횡단보도는 우선순위 높음 → QUEUE_FLUSH
            val qMode = if (status == "crosswalk") TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
            tts.speak(message, qMode, null, "surface-${SystemClock.elapsedRealtime()}")
        }

        // 차도 / 자전거도로: 5초마다 방향 유도 반복
        if (status == "road" || frontCls == "bike_lane") {
            val situation = if (frontCls == "bike_lane") "자전거 도로입니다." else "차도입니다."
            roadCtrl.start(situation, leftStatus, rightStatus)
        }
    }

    private fun startServerReconnectAlert() {
        stopServerReconnectAlert()
        val runnable = object : Runnable {
            override fun run() {
                tts.speak("서버에 연결할 수 없습니다. 주의하세요.",
                    TextToSpeech.QUEUE_ADD, null, "server-retry-${SystemClock.elapsedRealtime()}")
                serverReconnectHandler.postDelayed(this, 30_000)
            }
        }
        serverReconnectHandler.postDelayed(runnable, 30_000)
    }

    private fun stopServerReconnectAlert() {
        serverReconnectHandler.removeCallbacksAndMessages(null)
    }

    private fun showWarning(message: String, side: String, amplitude: Int = 180, speakNow: Boolean = false) {
        runOnUiThread {
            warningText.text = message
            warningBanner.visibility = View.VISIBLE
        }
        if (speakNow) {
            tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, "warn-${SystemClock.elapsedRealtime()}")
        }
        vibCtrl.vibrate(side, amplitude)
        warningBanner.postDelayed({ runOnUiThread { warningBanner.visibility = View.GONE } }, 3_000)
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 카메라 상태 (어두움 / 가림 / 손전등)
    // ══════════════════════════════════════════════════════════════════════════

    /** 비트맵 평균 밝기(V) 계산 — 8x8 다운샘플로 빠르게 */
    private fun avgBrightness(bmp: android.graphics.Bitmap): Int {
        val arr = thumbnail8(bmp)
        var sum = 0
        for (v in arr) sum += v
        return sum / arr.size
    }

    /** 8x8 다운샘플 명도 배열 — 장면 변화 감지·밝기 측정 공용 */
    private fun thumbnail8(bmp: android.graphics.Bitmap): IntArray {
        val w = 8; val h = 8
        val small = android.graphics.Bitmap.createScaledBitmap(bmp, w, h, true)
        val out = IntArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            val c = small.getPixel(x, y)
            out[y * w + x] = ((c shr 16 and 0xFF) + (c shr 8 and 0xFF) + (c and 0xFF)) / 3
        }
        small.recycle()
        return out
    }

    /** 두 8x8 썸네일의 평균 픽셀 차이 (0~255) */
    private fun thumbDiff(a: IntArray, b: IntArray): Int {
        if (a.size != b.size) return 255
        var sum = 0
        for (i in a.indices) sum += kotlin.math.abs(a[i] - b[i])
        return sum / a.size
    }

    private fun setTorch(on: Boolean) {
        val cm = cameraManager ?: return
        val id = torchCameraId ?: return
        if (on == torchOn) return
        runCatching {
            cm.setTorchMode(id, on)
            torchOn = on
        }.onFailure { Log.w(TAG, "손전등 제어 실패: ${it.message}") }
    }

    /** analyzeFrame에서 호출 — 밝기와 탐지 빈 결과 추적 */
    private fun trackCameraState(brightness: Int, anyDetectionOrSurface: Boolean) {
        val now = SystemClock.elapsedRealtime()
        val isDark = brightness < 30

        if (isDark) {
            if (darkSinceMs == 0L) darkSinceMs = now
            // 5초 이상 어두움 → 손전등 ON
            if (!torchOn && now - darkSinceMs >= 5_000L) {
                tts.speak("어두워서 잘 보이지 않습니다. 손전등을 켭니다.",
                    TextToSpeech.QUEUE_ADD, null, "torch-on")
                setTorch(true)
                torchDarkSinceMs = now
            }
            // 손전등 ON 후에도 5초 더 어두움 → 가림 안내 (10초마다 반복)
            if (torchOn && torchDarkSinceMs > 0L && now - torchDarkSinceMs >= 5_000L) {
                if (now - lastBlockSpokeMs >= 10_000L) {
                    tts.speak("카메라가 가려져 있거나 너무 어둡습니다. 카메라를 확인하세요.",
                        TextToSpeech.QUEUE_ADD, null, "cam-blocked-dark")
                    lastBlockSpokeMs = now
                }
            }
        } else {
            // 밝음 회복
            if (torchOn) {
                tts.speak("카메라가 정상 작동합니다.", TextToSpeech.QUEUE_ADD, null, "torch-off")
                setTorch(false)
            }
            darkSinceMs = 0L
            torchDarkSinceMs = 0L
        }

        // 가림 감지 (밝기는 정상인데 탐지·세그 둘 다 빈 결과)
        if (!isDark && !anyDetectionOrSurface) {
            emptyResultStreak++
            // 약 5초 (~10fps 가정 시 50프레임) 연속이면 가림 안내
            if (emptyResultStreak >= 50 && now - lastBlockSpokeMs >= 10_000L) {
                tts.speak("카메라가 가려진 것 같습니다.",
                    TextToSpeech.QUEUE_ADD, null, "cam-blocked")
                lastBlockSpokeMs = now
            }
        } else {
            emptyResultStreak = 0
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 화면 끄기 모드 (배터리 절감 5️⃣-A)
    //   - 검정 오버레이 풀스크린 + 백라이트 거의 OFF (screenBrightness 0.01)
    //   - Activity는 살아있음 → 카메라·탐지·음성·진동 정상 동작
    //   - FLAG_KEEP_SCREEN_ON은 유지 (자동 꺼짐으로 카메라 release 방지)
    //   - 오버레이 탭하면 해제
    // ══════════════════════════════════════════════════════════════════════════

    private fun toggleScreenOff(enable: Boolean) {
        if (enable == screenOffMode) return
        screenOffMode = enable
        runOnUiThread {
            if (enable) {
                // 백라이트 거의 OFF (0.0은 시스템 디폴트 반환이라 0.01 사용)
                window.attributes = window.attributes.apply { screenBrightness = 0.01f }
                screenOffOverlay.visibility = View.VISIBLE
                tts.speak("화면 끄기 모드입니다. 화면을 두 번 두드리면 다시 켜집니다.",
                    TextToSpeech.QUEUE_ADD, null, "screen-off")
            } else {
                // 시스템 자동 밝기로 복귀 (BRIGHTNESS_OVERRIDE_NONE = -1f)
                window.attributes = window.attributes.apply { screenBrightness = -1f }
                screenOffOverlay.visibility = View.GONE
                tts.speak("화면을 다시 켰습니다.",
                    TextToSpeech.QUEUE_ADD, null, "screen-on")
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 배터리 / 절전 모드
    // ══════════════════════════════════════════════════════════════════════════

    private val batteryReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action != Intent.ACTION_BATTERY_CHANGED) return
            val level  = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
            val scale  = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
            val status = intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
            val charging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                           status == BatteryManager.BATTERY_STATUS_FULL
            if (level < 0 || scale <= 0) return
            val pct = (level * 100) / scale
            handleBatteryLevel(pct, charging)
        }
    }

    private fun handleBatteryLevel(pct: Int, charging: Boolean) {
        // 충전 시작 안내 + 절전 모드 자동 해제
        if (charging && !isCharging) {
            isCharging = true
            tts.speak("충전이 시작되었습니다.", TextToSpeech.QUEUE_ADD, null, "charge-start")
            if (powerSaveMode) {
                powerSaveMode = false
                applyPowerSave(false)
            }
            return
        }
        if (!charging) isCharging = false

        // 임계 레벨 안내 (한 번씩만)
        when {
            pct <= 5 -> {
                val now = SystemClock.elapsedRealtime()
                if (now - lastBatteryCriticalSpokeMs >= 5_000L) {
                    tts.speak("배터리가 5퍼센트입니다. 곧 꺼집니다. 안전한 곳으로 이동하세요.",
                        TextToSpeech.QUEUE_FLUSH, null, "battery-critical")
                    lastBatteryCriticalSpokeMs = now
                }
                lastBatteryAlertLevel = 5
                if (!powerSaveMode) { powerSaveMode = true; applyPowerSave(true) }
            }
            pct <= 15 && lastBatteryAlertLevel > 15 -> {
                tts.speak("배터리가 15퍼센트 남았습니다. 절전 모드로 전환합니다.",
                    TextToSpeech.QUEUE_ADD, null, "battery-low15")
                lastBatteryAlertLevel = 15
                if (!powerSaveMode) { powerSaveMode = true; applyPowerSave(true) }
            }
            pct <= 30 && lastBatteryAlertLevel > 30 -> {
                tts.speak("배터리가 30퍼센트 남았습니다.",
                    TextToSpeech.QUEUE_ADD, null, "battery-low30")
                lastBatteryAlertLevel = 30
            }
            pct > 30 -> lastBatteryAlertLevel = 100  // 충전 등으로 복구되면 리셋
        }
    }

    /** 절전 모드 진입/해제에 따른 설정 */
    private fun applyPowerSave(enable: Boolean) {
        runOnUiThread {
            if (enable) {
                window.clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            } else {
                window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
        }
        // 나침반 센서 주기 조정
        rotationSensor?.let {
            sensorManager.unregisterListener(compassListener)
            val rate = if (enable) SensorManager.SENSOR_DELAY_NORMAL else SensorManager.SENSOR_DELAY_UI
            sensorManager.registerListener(compassListener, it, rate)
        }
        // 차도/자전거도로 반복 안내 간격 (정상 5초 / 절전 10초)
        roadCtrl.setInterval(if (enable) 10_000L else 5_000L)
        Log.i(TAG, "절전 모드: $enable")
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 진동
    // ══════════════════════════════════════════════════════════════════════════

    // ══════════════════════════════════════════════════════════════════════════
    // TTS
    // ══════════════════════════════════════════════════════════════════════════

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.KOREAN
            tts.speak(
                "SafeStep 시작됩니다. 마이크 버튼을 눌러 목적지를 말씀해주세요.",
                TextToSpeech.QUEUE_FLUSH, null, "intro"
            )
            checkInitialStates()
        }
    }
}
