package com.safestep.app

import android.content.Intent
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.widget.Button
import android.widget.EditText
import androidx.appcompat.app.AppCompatActivity
import com.safestep.app.assistant.AssistantAction
import com.safestep.app.assistant.VoiceAssistant
import com.safestep.app.detect.RemoteDetector
import java.util.Locale

class SplashActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var tts: TextToSpeech
    private lateinit var serverUrlInput: EditText
    private lateinit var voiceAssistant: VoiceAssistant

    companion object {
        const val PREFS_NAME = "safestep_prefs"
        const val KEY_SERVER_URL = "server_url"
        const val DEFAULT_SERVER_URL = "https://unglued-grill-unlovely.ngrok-free.dev/detect"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_splash)

        tts = TextToSpeech(this, this)

        // 서버 URL 입력창 — 저장된 값 불러오기
        serverUrlInput = findViewById(R.id.serverUrlInput)
        val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        val savedUrl = prefs.getString(KEY_SERVER_URL, DEFAULT_SERVER_URL) ?: DEFAULT_SERVER_URL
        serverUrlInput.setText(savedUrl)
        // 앱 시작 시 SharedPreferences 값을 RemoteDetector에 즉시 반영
        RemoteDetector.SERVER_URL = savedUrl

        // 카메라 모드 → MapActivity
        findViewById<Button>(R.id.startButton).setOnClickListener {
            saveServerUrl()
            tts.stop()
            startActivity(Intent(this, MapActivity::class.java))
            finish()
        }

        // 동영상 테스트 모드 → VideoTestActivity
        findViewById<Button>(R.id.videoTestButton).setOnClickListener {
            saveServerUrl()
            tts.stop()
            startActivity(Intent(this, VideoTestActivity::class.java))
            finish()
        }

        // 음성 어시스턴트 — 화면 길게 누르면 활성화
        val baseUrl = savedUrl.trimEnd('/').removeSuffix("/detect")
        voiceAssistant = VoiceAssistant(
            activity      = this,
            serverBaseUrl = baseUrl,
            screen        = "splash",
            onAction      = { action -> handleAssistantAction(action) },
        )
        voiceAssistant.attachLongPressTo(findViewById(android.R.id.content))
    }

    private fun handleAssistantAction(action: AssistantAction) {
        when (action) {
            is AssistantAction.EnterCameraMode -> {
                saveServerUrl()
                tts.stop()
                startActivity(Intent(this, MapActivity::class.java))
                finish()
            }
            is AssistantAction.EnterVideoTestMode -> {
                saveServerUrl()
                tts.stop()
                startActivity(Intent(this, VideoTestActivity::class.java))
                finish()
            }
            else -> { /* SpeakOnly 등은 VoiceAssistant가 자동으로 TTS 발화 */ }
        }
    }

    /** EditText 값을 SharedPreferences에 저장 */
    private fun saveServerUrl() {
        val url = serverUrlInput.text.toString().trim().trimEnd('/')
        if (url.isNotEmpty()) {
            getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
                .edit()
                .putString(KEY_SERVER_URL, url)
                .apply()
            // RemoteDetector에 즉시 반영 — VideoTestActivity는 SharedPreferences를
            // 읽지 않고 SERVER_URL을 바로 사용하기 때문에 여기서 함께 설정해야 함
            RemoteDetector.SERVER_URL = url
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.KOREAN
            tts.speak(
                "SafeStep. 카메라 모드 또는 동영상 테스트 모드를 선택해주세요.",
                TextToSpeech.QUEUE_FLUSH, null, "splash-intro"
            )
        }
    }

    override fun onDestroy() {
        tts.shutdown()
        voiceAssistant.release()
        super.onDestroy()
    }
}
