package com.safestep.app.navigation

import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import kotlin.math.*

/**
 * 경로 스텝 목록 + GPS 위치 업데이트를 받아
 * 턴바이턴 TTS 안내를 제공한다.
 */
class NavigationGuide(private val tts: TextToSpeech) {

    private val TAG = "NavGuide"

    private var steps = listOf<RouteStep>()
    private var currentIdx = 0
    private var isActive = false
    private var lastAnnouncedIdx = -1
    private var lastAnnounceMs = 0L

    /** 스텝이 바뀔 때 UI 갱신용 콜백 */
    var onStepChanged: ((RouteStep) -> Unit)? = null
    /** 목적지 도착 콜백 */
    var onArrived: (() -> Unit)? = null
    /** 경로 이탈 콜백 */
    var onOffRoute: (() -> Unit)? = null
    /** 반대 방향으로 걷는 중 콜백 */
    var onWrongDirection: (() -> Unit)? = null

    companion object {
        private const val STEP_ARRIVE_M        = 20.0   // 이 거리 이내면 현재 스텝 통과
        private const val ANNOUNCE_PRE_M       = 30.0   // 다음 회전 미리 안내 거리
        private const val COOLDOWN_MS          = 8_000L // 같은 안내 반복 최소 간격
        private const val OFF_ROUTE_M          = 50.0   // 이 거리 초과 시 경로 이탈 판정
        private const val OFF_ROUTE_COOLDOWN_MS     = 15_000L
        private const val WRONG_DIR_MIN_MOVE_M      = 5.0    // 반대 방향 감지용 최소 이동 거리
        private const val WRONG_DIR_ANGLE_DEG       = 120.0  // 이 각도 이상 벗어나면 반대 방향
        private const val WRONG_DIR_COOLDOWN_MS     = 10_000L
    }

    private var lastOffRouteMs = 0L
    private var lastWrongDirMs = 0L
    private var prevLat: Double? = null
    private var prevLon: Double? = null

    fun start(routeSteps: List<RouteStep>) {
        steps = routeSteps
        currentIdx = 0
        lastAnnouncedIdx = -1
        lastOffRouteMs = 0L
        lastWrongDirMs = 0L
        prevLat = null
        prevLon = null
        isActive = true
        announce(0)
        onStepChanged?.invoke(steps[0])
    }

    fun stop() {
        isActive = false
    }

    fun isRunning() = isActive

    /** GPS 재연결 시 호출 — 이탈 쿨다운을 초기화해 즉시 재탐색 가능하게 함 */
    fun resetOffRouteCooldown() {
        lastOffRouteMs = 0L
    }

    /** GPS 위치가 업데이트될 때마다 호출 */
    fun updateLocation(lat: Double, lon: Double) {
        if (!isActive || steps.isEmpty()) return

        // 목적지 도착 확인
        val dest = steps.last()
        if (distM(lat, lon, dest.lat, dest.lon) < STEP_ARRIVE_M) {
            isActive = false
            tts.speak("목적지에 도착했습니다.", TextToSpeech.QUEUE_FLUSH, null, "arrived")
            onArrived?.invoke()
            return
        }

        // 현재 스텝 지점 통과 → 다음 스텝으로 전진
        if (currentIdx < steps.size - 1) {
            val cur = steps[currentIdx]
            val distToCur = distM(lat, lon, cur.lat, cur.lon)
            if (distToCur < STEP_ARRIVE_M) {
                currentIdx++
                lastOffRouteMs = 0L  // 스텝 통과 시 이탈 타이머 초기화
                announce(currentIdx)
                onStepChanged?.invoke(steps[currentIdx])
                return
            }

            // 경로 이탈 감지
            val now = SystemClock.elapsedRealtime()
            if (distToCur > OFF_ROUTE_M && now - lastOffRouteMs > OFF_ROUTE_COOLDOWN_MS) {
                lastOffRouteMs = now
                Log.d(TAG, "경로 이탈 감지: 현재 스텝까지 ${distToCur.toInt()}m")
                onOffRoute?.invoke()
                return
            }
        }

        // 다음 스텝까지 ANNOUNCE_PRE_M 이내 → 미리 안내
        val nextIdx = currentIdx + 1
        if (nextIdx < steps.size) {
            val next = steps[nextIdx]
            val dist = distM(lat, lon, next.lat, next.lon)
            val now  = SystemClock.elapsedRealtime()
            if (dist < ANNOUNCE_PRE_M &&
                nextIdx != lastAnnouncedIdx &&
                now - lastAnnounceMs > COOLDOWN_MS
            ) {
                announce(nextIdx)
            }
        }

        // ── 반대 방향 감지 ───────────────────────────────────────────────
        val pLat = prevLat
        val pLon = prevLon
        if (pLat != null && pLon != null) {
            val moveDist = distM(pLat, pLon, lat, lon)
            if (moveDist >= WRONG_DIR_MIN_MOVE_M) {
                val moveBearing  = bearing(pLat, pLon, lat, lon)
                val routeBearing = bearing(lat, lon, steps[currentIdx].lat, steps[currentIdx].lon)
                val angleDiff    = angleDiff(moveBearing, routeBearing)
                val now = SystemClock.elapsedRealtime()
                if (angleDiff > WRONG_DIR_ANGLE_DEG && now - lastWrongDirMs > WRONG_DIR_COOLDOWN_MS) {
                    lastWrongDirMs = now
                    onWrongDirection?.invoke()
                }
            }
        }
        prevLat = lat
        prevLon = lon
    }

    /** 현재 스텝 (UI 표시용) */
    fun currentStep(): RouteStep? = steps.getOrNull(currentIdx)

    /** 현재 위치 → 현재 스텝 지점까지 거리 (m) */
    fun distToCurrentStep(lat: Double, lon: Double): Int {
        val s = steps.getOrNull(currentIdx) ?: return 0
        return distM(lat, lon, s.lat, s.lon).toInt()
    }

    private fun announce(idx: Int) {
        val step = steps.getOrNull(idx) ?: return
        if (step.description.isBlank()) return
        lastAnnouncedIdx = idx
        lastAnnounceMs   = SystemClock.elapsedRealtime()
        Log.d(TAG, "안내[$idx] ${step.description}")
        tts.speak(step.description, TextToSpeech.QUEUE_ADD, null, "nav-$idx")
    }

    /** 방위각 계산 (0~360도, 북=0, 동=90) */
    private fun bearing(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val dLon = Math.toRadians(lon2 - lon1)
        val lat1R = Math.toRadians(lat1)
        val lat2R = Math.toRadians(lat2)
        val y = sin(dLon) * cos(lat2R)
        val x = cos(lat1R) * sin(lat2R) - sin(lat1R) * cos(lat2R) * cos(dLon)
        return (Math.toDegrees(atan2(y, x)) + 360) % 360
    }

    /** 두 방위각의 차이 (0~180도) */
    private fun angleDiff(a: Double, b: Double): Double {
        val diff = abs(a - b) % 360
        return if (diff > 180) 360 - diff else diff
    }

    /** Haversine 거리 (m) */
    private fun distM(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val R    = 6_371_000.0
        val dLat = Math.toRadians(lat2 - lat1)
        val dLon = Math.toRadians(lon2 - lon1)
        val a    = sin(dLat / 2).pow(2) +
                   cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) * sin(dLon / 2).pow(2)
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))
    }
}
