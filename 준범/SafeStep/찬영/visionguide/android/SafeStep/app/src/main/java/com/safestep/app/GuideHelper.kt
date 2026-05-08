package com.safestep.app

import android.os.Handler
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.tts.TextToSpeech

// ══════════════════════════════════════════════════════════════════════════
// 두 Activity (MapActivity / VideoTestActivity) 공용 안내 헬퍼.
//   - 순수 함수: groupNoun, buildDirectionGuide, surfaceMessageFor
//   - 상태 보유 클래스: VibrationController, RoadRepeatController
// ══════════════════════════════════════════════════════════════════════════

/** 그룹명 → 짧은 명사 (음성 길이 제한용) */
fun groupNoun(group: String): String = when (group) {
    "차량"          -> "차량"
    "개인이동장치"  -> "자전거"
    "사람/동물"     -> "사람"
    "고정장애물"    -> "장애물"
    "신호등/표지판" -> "표지"
    else            -> group
}

/**
 * 차도·자전거도로 5초 반복 시 양옆 노면을 보고 동적 유도 음성 생성.
 * `상황별 반응.md`의 방향 유도 표 참고.
 */
fun buildDirectionGuide(situation: String, leftStatus: String, rightStatus: String): String {
    val leftSafe   = leftStatus == "sidewalk"
    val rightSafe  = rightStatus == "sidewalk"
    val leftAlley  = leftStatus == "alley"
    val rightAlley = rightStatus == "alley"
    val leftCross  = leftStatus == "crosswalk"
    val rightCross = rightStatus == "crosswalk"
    return when {
        leftCross || rightCross -> {
            val dir = if (leftCross) "왼쪽" else "오른쪽"
            "$situation $dir 횡단보도 방향으로 이동하세요."
        }
        leftSafe && rightSafe   -> "$situation 왼쪽 또는 오른쪽 인도로 이동하세요."
        leftSafe                -> "$situation 왼쪽 인도로 이동하세요."
        rightSafe               -> "$situation 오른쪽 인도로 이동하세요."
        leftAlley && rightAlley -> "$situation 왼쪽 또는 오른쪽 골목 방향으로 이동하세요."
        leftAlley               -> "$situation 왼쪽 골목 방향으로 이동하세요."
        rightAlley              -> "$situation 오른쪽 골목 방향으로 이동하세요."
        else                    -> "$situation 천천히 멈추고 주변을 확인하세요."
    }
}

/**
 * 노면 frontCls / status → 안내 음성 매핑. 없으면 null.
 * `상황별 반응.md`의 노면 표 참고.
 */
fun surfaceMessageFor(frontCls: String, status: String): String? = when {
    frontCls == "sidewalk_damaged"             -> "파손된 인도입니다. 주의하세요."
    frontCls == "braille_guide_blocks_damaged" -> "파손된 점자블록입니다. 주의하세요."
    frontCls == "caution_zone_grating"         -> "격자 덮개입니다. 주의하세요."
    frontCls == "caution_zone_manhole"         -> "맨홀이 있습니다. 주의하세요."
    frontCls == "caution_zone_repair_zone"     -> "공사 구역입니다. 주의하세요."
    frontCls == "caution_zone_stairs"          -> "계단입니다. 주의하세요."
    frontCls == "caution_zone_tree_zone"       -> "나무 구역입니다. 주의하세요."
    frontCls == "bike_lane"                    -> "자전거 도로입니다. 주의하세요."
    status == "crosswalk"                      -> "횡단보도입니다. 멈추고 신호를 확인하세요."
    status == "road"                           -> "차도입니다. 주의하세요."
    status == "alley"                          -> "골목길입니다. 주의하세요."
    status == "sidewalk"                       -> "인도입니다."
    else                                       -> null
}

// ══════════════════════════════════════════════════════════════════════════
// 진동 컨트롤러
//   - 1초 슬롯 코얼레싱 (진행 중 약한 진동 무시, 강은 약·중 끊고 즉시)
//   - 100ms 윈도우 코얼레싱 (짧은 시간에 여러 호출 오면 가장 강한 amplitude만)
//   - 긴급(255)/일반 패턴 분리
// ══════════════════════════════════════════════════════════════════════════
class VibrationController(private val vibrator: Vibrator?) {
    @Volatile private var busyUntilMs = 0L
    @Volatile private var currentAmp  = 0
    private val urgentSlotMs = 1_000L
    private val normalSlotMs = 500L

    // 100ms 코얼레싱: 같은 윈도우 내 호출은 가장 강한 것만 실행
    private val coalesceWindowMs = 100L
    private val coalesceHandler  = Handler(android.os.Looper.getMainLooper())
    private var pendingSide: String? = null
    private var pendingAmp:  Int     = 0
    private val flushRunnable = Runnable {
        val side = pendingSide ?: return@Runnable
        val amp  = pendingAmp
        pendingSide = null; pendingAmp = 0
        actuallyVibrate(side, amp)
    }

    /**
     * 진동 요청. 100ms 윈도우 안에 여러 호출 오면 가장 강한 것만 실제 실행.
     * @param side 왼쪽/오른쪽/그 외
     * @param amplitude 0~255 (255=긴급)
     */
    fun vibrate(side: String, amplitude: Int = 180) {
        // 더 약한 호출은 무시 (윈도우 안 더 강한 게 이미 예약됨)
        if (pendingSide != null && amplitude <= pendingAmp) return
        pendingSide = side
        pendingAmp  = amplitude
        coalesceHandler.removeCallbacks(flushRunnable)
        coalesceHandler.postDelayed(flushRunnable, coalesceWindowMs)
    }

    private fun actuallyVibrate(side: String, amplitude: Int) {
        val now = SystemClock.elapsedRealtime()
        if (now < busyUntilMs && amplitude <= currentAmp) return

        // 긴급(255)만 긴 패턴, 일반은 짧은 패턴(~0.4초) — 진동 모터 전력 절감
        val timings = if (amplitude >= 255) {
            when (side) {
                "왼쪽"  -> longArrayOf(0, 120, 80, 120, 80, 350)
                "오른쪽" -> longArrayOf(0, 350, 80, 120, 80, 120)
                else    -> longArrayOf(0, 250, 80, 100, 80, 250)
            }
        } else {
            when (side) {
                "왼쪽"  -> longArrayOf(0, 60,  60, 200)
                "오른쪽" -> longArrayOf(0, 200, 60, 60)
                else    -> longArrayOf(0, 150, 60, 150)
            }
        }
        val amplitudes = IntArray(timings.size) { i -> if (i % 2 == 0) 0 else amplitude }

        if (now < busyUntilMs) vibrator?.cancel()
        vibrator?.vibrate(VibrationEffect.createWaveform(timings, amplitudes, -1))
        busyUntilMs = now + (if (amplitude >= 255) urgentSlotMs else normalSlotMs)
        currentAmp  = amplitude
    }
}

// ══════════════════════════════════════════════════════════════════════════
// 차도/자전거도로 5초 반복 안내 컨트롤러
//   - 절전 모드 진입 시 setInterval(10_000)로 간격 늘릴 수 있음
// ══════════════════════════════════════════════════════════════════════════
class RoadRepeatController(
    private val tts: TextToSpeech,
    private val handler: Handler,
    initialIntervalMs: Long = 5_000L,
) {
    @Volatile private var running = false
    @Volatile private var intervalMs = initialIntervalMs
    // 마지막 start 인자 보관 — 간격 변경 시 재시작에 사용
    private var lastSituation = ""
    private var lastLeft = ""
    private var lastRight = ""

    fun start(situation: String, leftStatus: String, rightStatus: String) {
        lastSituation = situation
        lastLeft  = leftStatus
        lastRight = rightStatus
        running = true
        val runnable = object : Runnable {
            override fun run() {
                if (!running) return
                val guide = buildDirectionGuide(situation, leftStatus, rightStatus)
                tts.speak(guide, TextToSpeech.QUEUE_ADD, null,
                    "road-repeat-${SystemClock.elapsedRealtime()}")
                handler.postDelayed(this, intervalMs)
            }
        }
        handler.postDelayed(runnable, intervalMs)
    }

    fun stop() {
        running = false
        handler.removeCallbacksAndMessages(null)
    }

    /** 절전 모드 진입/해제에 따라 반복 간격 변경. 동작 중이면 재시작. */
    fun setInterval(ms: Long) {
        if (intervalMs == ms) return
        intervalMs = ms
        if (running) {
            stop()
            start(lastSituation, lastLeft, lastRight)
        }
    }
}
