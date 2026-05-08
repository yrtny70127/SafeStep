package com.safestep.app.navigation

import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import kotlin.math.*

/**
 * 경로 스텝 목록 + GPS 위치 업데이트를 받아
 * 턴바이턴 TTS 안내를 제공한다.
 *
 * 예고 안내 3단계:
 *   150m → "약 150m 앞, [안내]"
 *    60m → "60m 앞, [안내]"
 *    현장 → "[안내]" (STEP_ARRIVE_M 이내, 즉시)
 */
class NavigationGuide(private val tts: TextToSpeech) {

    private val TAG = "NavGuide"

    private var steps = listOf<RouteStep>()
    private var currentIdx = 0
    private var isActive = false

    /** 현재 스텝의 목표 지점(다음 회전)에 대해 이미 발화한 거리 임계값 집합 */
    private val announcedDistances = mutableSetOf<Int>()

    /** true: start() 또는 스텝 전진 직후 — 이미 지나친 임계값을 조용히 채워 오발화 방지 */
    private var justAdvanced = false

    /** 목적지 근접(50m) 안내 발화 여부 */
    private var announcedNearDest = false

    /** 스텝이 바뀔 때 UI 갱신용 콜백 */
    var onStepChanged: ((RouteStep) -> Unit)? = null
    /** 목적지 도착 콜백 */
    var onArrived: (() -> Unit)? = null

    companion object {
        /** 이 거리 이내면 현재 스텝 지점 통과 → 다음 스텝 전진 */
        private const val STEP_ARRIVE_M = 20.0
        /**
         * 예고 임계값 (m).
         * 각 값 이하로 진입 시 해당 단계 안내를 한 번 발화.
         * 내림차순 정렬 필수 → 멀리서부터 순서대로 announce.
         */
        private val PROX_THRESHOLDS = listOf(150, 60)
    }

    // ─────────────────────────────────────────────────────────────────────────

    fun start(routeSteps: List<RouteStep>) {
        steps              = routeSteps
        currentIdx         = 0
        isActive           = true
        announcedNearDest  = false
        announcedDistances.clear()
        justAdvanced       = true   // 첫 위치 업데이트에서 이미 지나친 임계값 조용히 채움
        announce(0)
        onStepChanged?.invoke(steps[0])
    }

    fun stop() { isActive = false }

    fun isRunning() = isActive

    /** GPS 위치가 업데이트될 때마다 호출 */
    fun updateLocation(lat: Double, lon: Double) {
        if (!isActive || steps.isEmpty()) return

        // ① 목적지 도착 확인
        val dest     = steps.last()
        val distDest = distM(lat, lon, dest.lat, dest.lon)

        if (distDest < STEP_ARRIVE_M) {
            isActive = false
            tts.speak("목적지에 도착했습니다.", TextToSpeech.QUEUE_FLUSH, null, "arrived")
            onArrived?.invoke()
            return
        }

        // 목적지 50m 근접 안내 (1회)
        if (!announcedNearDest && distDest < 50.0) {
            announcedNearDest = true
            tts.speak("목적지가 약 ${distDest.toInt()}m 앞입니다.",
                TextToSpeech.QUEUE_ADD, null, "near-dest")
        }

        // ② 현재 스텝 지점 통과 → 다음 스텝 전진
        if (currentIdx < steps.size - 1) {
            val cur = steps[currentIdx]
            if (distM(lat, lon, cur.lat, cur.lon) < STEP_ARRIVE_M) {
                currentIdx++
                announcedDistances.clear()          // 새 스텝 → 예고 초기화
                justAdvanced = true                 // 이미 지나친 임계값 오발화 방지
                announce(currentIdx)                // 즉시 현재 안내
                onStepChanged?.invoke(steps[currentIdx])
                return
            }
        }

        // ③ 다음 스텝(회전 지점)까지의 거리 기반 3단계 예고 안내
        //    현재 스텝 지점 = 다음 회전 지점
        val nextTurnIdx = currentIdx   // 현재 스텝의 목표 = 다음 회전
        val distToTurn = distM(lat, lon, steps[nextTurnIdx].lat, steps[nextTurnIdx].lon)

        // start() 또는 스텝 전진 직후 첫 업데이트:
        // 이미 지나친 임계값을 조용히 채워 소급 오발화 방지
        // (예: 첫 웨이포인트가 50m 앞일 때 "150m 앞" 오발화 막기)
        if (justAdvanced) {
            justAdvanced = false
            for (thresh in PROX_THRESHOLDS) {
                if (distToTurn < thresh) announcedDistances.add(thresh)
            }
        }

        // 내림차순으로 순회 → 150m 먼저, 가까워질수록 60m
        for (thresh in PROX_THRESHOLDS.sortedDescending()) {
            if (distToTurn < thresh && thresh !in announcedDistances) {
                announcedDistances.add(thresh)
                val msg = "${thresh}m 앞, ${steps[nextTurnIdx].description}"
                tts.speak(msg, TextToSpeech.QUEUE_ADD, null, "prox-$thresh-$nextTurnIdx")
                Log.d(TAG, "예고[$thresh m] ${steps[nextTurnIdx].description}")
                break   // 한 GPS 업데이트에 하나씩만 발화
            }
        }
    }

    /** 현재 스텝 (UI 표시용) */
    fun currentStep(): RouteStep? = steps.getOrNull(currentIdx)

    /** 현재 스텝 번호 (1-based, UI 카운터용) */
    fun currentStepNumber(): Int = currentIdx + 1

    /** 현재 위치 → 현재 스텝 지점까지 거리 (m) */
    fun distToCurrentStep(lat: Double, lon: Double): Int {
        val s = steps.getOrNull(currentIdx) ?: return 0
        return distM(lat, lon, s.lat, s.lon).toInt()
    }

    // ─────────────────────────────────────────────────────────────────────────

    private fun announce(idx: Int) {
        val step = steps.getOrNull(idx) ?: return
        if (step.description.isBlank()) return
        Log.d(TAG, "안내[$idx] ${step.description}")
        tts.speak(step.description, TextToSpeech.QUEUE_ADD, null, "nav-$idx")
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
