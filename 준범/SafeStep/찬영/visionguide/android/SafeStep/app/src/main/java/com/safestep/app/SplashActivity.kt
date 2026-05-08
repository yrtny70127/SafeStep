package com.safestep.app

import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.safestep.app.detect.RemoteDetector
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale

class SplashActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var tts: TextToSpeech

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_splash)

        tts = TextToSpeech(this, this)

        // SharedPreferences에서 저장된 URL 불러오기
        val prefs = getSharedPreferences(RemoteDetector.PREF_NAME, Context.MODE_PRIVATE)
        val savedUrl = prefs.getString(RemoteDetector.PREF_SERVER_URL, "") ?: ""
        if (savedUrl.isNotEmpty()) {
            RemoteDetector.SERVER_URL = savedUrl
        }

        // 카메라 모드 → MapActivity
        findViewById<Button>(R.id.startButton).setOnClickListener {
            if (RemoteDetector.SERVER_URL.isEmpty() || RemoteDetector.SERVER_URL.startsWith("http://192")) {
                showUrlDialog { tts.stop(); startActivity(Intent(this, MapActivity::class.java)); finish() }
            } else {
                tts.stop()
                startActivity(Intent(this, MapActivity::class.java))
                finish()
            }
        }

        // 동영상 테스트 모드 → VideoTestActivity
        findViewById<Button>(R.id.videoTestButton).setOnClickListener {
            if (RemoteDetector.SERVER_URL.isEmpty() || RemoteDetector.SERVER_URL.startsWith("http://192")) {
                showUrlDialog { tts.stop(); startActivity(Intent(this, VideoTestActivity::class.java)); finish() }
            } else {
                tts.stop()
                startActivity(Intent(this, VideoTestActivity::class.java))
                finish()
            }
        }
    }

    /**
     * ngrok URL 입력 다이얼로그.
     * 확인 후 SharedPreferences 저장 → onConfirmed 콜백 실행.
     */
    private fun showUrlDialog(onConfirmed: () -> Unit) {
        val prefs = getSharedPreferences(RemoteDetector.PREF_NAME, Context.MODE_PRIVATE)
        val currentUrl = prefs.getString(RemoteDetector.PREF_SERVER_URL, "") ?: ""

        val input = EditText(this).apply {
            hint = "https://xxxx.ngrok-free.app"
            setText(currentUrl)
            setPadding(48, 24, 48, 24)
        }

        AlertDialog.Builder(this)
            .setTitle("서버 URL 설정")
            .setMessage("ngrok URL을 입력하세요.\n(예: https://xxxx.ngrok-free.app)")
            .setView(input)
            .setPositiveButton("확인") { _, _ ->
                val url = input.text.toString().trim().trimEnd('/')
                if (url.isEmpty()) {
                    Toast.makeText(this, "URL을 입력해주세요", Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }
                // /detect 경로 자동 추가
                val detectUrl = if (url.endsWith("/detect")) url else "$url/detect"
                RemoteDetector.SERVER_URL = detectUrl
                prefs.edit().putString(RemoteDetector.PREF_SERVER_URL, detectUrl).apply()
                Toast.makeText(this, "서버 설정 완료", Toast.LENGTH_SHORT).show()
                onConfirmed()
            }
            .setNegativeButton("취소", null)
            .show()
    }

    override fun onResume() {
        super.onResume()
        checkServerStatus()
    }

    private fun checkServerStatus() {
        val baseUrl = RemoteDetector.SERVER_URL
            .removeSuffix("/detect")
            .trimEnd('/')
        if (baseUrl.isEmpty()) {
            updateServerDot(false)
            return
        }
        Thread {
            val ok = try {
                val conn = URL("$baseUrl/health").openConnection() as HttpURLConnection
                conn.connectTimeout = 3_000
                conn.readTimeout    = 3_000
                conn.responseCode == 200
            } catch (e: Exception) { false }
            runOnUiThread { updateServerDot(ok) }
        }.start()
    }

    private fun updateServerDot(connected: Boolean) {
        val color = if (connected) Color.parseColor("#34C759") else Color.parseColor("#FF3B30")
        val dot   = findViewById<View>(R.id.serverDot)
        val text  = findViewById<TextView>(R.id.serverStatusText)
        dot.background = GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(color)
        }
        text.setTextColor(color)
        text.text = if (connected) "서버 연결됨" else "서버 끊김"
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
        super.onDestroy()
    }
}
