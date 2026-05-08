package com.safestep.app.assistant

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Vibrator
import android.os.VibrationEffect
import android.os.Build
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.View
import androidx.core.app.ActivityCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

/**
 * SafeStep 음성 어시스턴트.
 *
 * 동작:
 *   1. attachLongPressTo(view) 로 화면에 부착
 *   2. 화면 1초 길게 누르면 → 진동 + SpeechRecognizer 시작
 *   3. 인식된 텍스트를 서버 /assistant 로 전송 (screen 컨텍스트 포함)
 *   4. 받은 AssistantAction 을 onAction 콜백으로 전달
 *   5. action.tts 를 TTS 로 자동 발화
 *
 * @param activity        호스트 액티비티 (권한 체크용)
 * @param serverBaseUrl   서버 base URL (예: "https://xxx.ngrok.io")
 * @param screen          현재 화면 식별자 — "splash" 또는 "map".
 *                        서버가 화면별로 다른 도구 세트를 LLM 에 바인딩합니다.
 * @param onAction        action 처리 콜백 (UI 스레드에서 호출됨)
 */
class VoiceAssistant(
    private val activity: Activity,
    serverBaseUrl: String,
    private val screen: String,
    private val onAction: (AssistantAction) -> Unit,
) {
    private val client       = AssistantClient(serverBaseUrl)
    private val scope        = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private val recognizer   = SpeechRecognizer.createSpeechRecognizer(activity)
    private val tts: TextToSpeech
    private val vibrator     = activity.getSystemService(Vibrator::class.java)

    /** 외부에서 주기적으로 갱신해주는 앱 컨텍스트. */
    private var context: Map<String, Any?> = emptyMap()

    private var inFlight: Job? = null
    private var ttsReady = false

    init {
        tts = TextToSpeech(activity) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts.language = Locale.KOREAN
                ttsReady = true
            } else {
                Log.e(TAG, "TTS 초기화 실패")
            }
        }
        recognizer.setRecognitionListener(buildListener())
    }

    /** 컨텍스트 업데이트 — MapActivity 에서 위치/내비 상태 변할 때 호출. */
    fun updateContext(
        isNavigating: Boolean? = null,
        lat: Double? = null,
        lon: Double? = null,
        remainingDistanceM: Int? = null,
        recentDetections: List<String>? = null,
    ) {
        context = buildMap {
            isNavigating?.let       { put("is_navigating", it) }
            lat?.let                { put("current_lat", it) }
            lon?.let                { put("current_lon", it) }
            remainingDistanceM?.let { put("remaining_distance_m", it) }
            recentDetections?.let   { put("recent_detections", it) }
        }
    }

    /** 화면(또는 특정 뷰)에 long press 리스너 부착. */
    fun attachLongPressTo(view: View) {
        val detector = GestureDetector(activity, object : GestureDetector.SimpleOnGestureListener() {
            override fun onLongPress(e: MotionEvent) {
                onLongPressTriggered()
            }
        })
        view.setOnTouchListener { v, ev ->
            detector.onTouchEvent(ev)
            v.performClick()
            false  // 다른 터치 핸들러 동작 방해 안 함
        }
    }

    /** 액티비티 onDestroy 에서 호출. */
    fun release() {
        scope.cancel()
        recognizer.destroy()
        if (ttsReady) tts.stop()
        tts.shutdown()
    }

    // ────────────────────────────────────────────────────
    // 내부
    // ────────────────────────────────────────────────────

    private fun onLongPressTriggered() {
        if (!hasMicPermission()) {
            ActivityCompat.requestPermissions(
                activity, arrayOf(Manifest.permission.RECORD_AUDIO), REQ_MIC,
            )
            return
        }
        if (inFlight?.isActive == true) {
            Log.d(TAG, "이미 처리 중 — 무시")
            return
        }
        vibrate(100)
        startListening()
    }

    private fun hasMicPermission(): Boolean =
        ActivityCompat.checkSelfPermission(activity, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED

    private fun startListening() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(
                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM,
            )
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
        }
        recognizer.startListening(intent)
    }

    private fun buildListener() = object : RecognitionListener {
        override fun onReadyForSpeech(params: Bundle?) {
            Log.d(TAG, "듣기 시작")
        }
        override fun onResults(results: Bundle?) {
            val texts = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
            val text  = texts?.firstOrNull().orEmpty()
            if (text.isBlank()) {
                speak("다시 말씀해주세요.")
                return
            }
            Log.d(TAG, "인식: $text")
            askServer(text)
        }
        override fun onError(error: Int) {
            Log.w(TAG, "음성 인식 에러: $error")
            speak("다시 말씀해주세요.")
        }
        override fun onBeginningOfSpeech() {}
        override fun onEndOfSpeech()       { vibrate(50) }
        override fun onRmsChanged(rmsdB: Float) {}
        override fun onBufferReceived(buffer: ByteArray?) {}
        override fun onPartialResults(partialResults: Bundle?) {}
        override fun onEvent(eventType: Int, params: Bundle?) {}
    }

    private fun askServer(text: String) {
        // 매 요청에 screen 컨텍스트를 자동 포함 — 서버가 화면별 도구 세트를 사용
        val ctx = context + mapOf("screen" to screen)
        inFlight = scope.launch {
            val action = try {
                withContext(Dispatchers.IO) { client.ask(text, ctx) }
            } catch (e: Exception) {
                Log.e(TAG, "서버 호출 실패", e)
                speak("서버에 연결할 수 없습니다.")
                return@launch
            }
            // tts 발화 + 앱 핸들러 호출 (둘 다 UI 스레드)
            if (action.tts.isNotBlank()) speak(action.tts)
            onAction(action)
        }
    }

    private fun speak(text: String) {
        if (!ttsReady) return
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "assistant")
    }

    @Suppress("DEPRECATION")
    private fun vibrate(ms: Long) {
        val v = vibrator ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            v.vibrate(VibrationEffect.createOneShot(ms, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            v.vibrate(ms)
        }
    }

    companion object {
        private const val TAG     = "VoiceAssistant"
        private const val REQ_MIC = 9201
    }
}
