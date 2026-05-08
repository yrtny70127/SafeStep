package com.safestep.app.detect

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.util.Log
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

/**
 * 서버에 프레임을 전송하고 탐지 결과를 받아오는 디텍터.
 *
 * ★ 서버 IP가 바뀌면 SERVER_URL 만 수정하면 됩니다.
 */
class RemoteDetector : ObjectDetector {

    companion object {
        private const val TAG = "RemoteDetector"

        // ★ SplashActivity에서 SharedPreferences로 설정됩니다
        // 기본값: 빈 문자열 (앱 시작 시 URL 입력 요청)
        var SERVER_URL = "http://192.168.75.218:8000/detect"

        /** 전송할 JPEG 압축 품질 (0~100). 낮을수록 빠르지만 정확도 저하.
         *  ★ SegmentationClient.JPEG_QUALITY와 반드시 동일해야 서버측 추론 캐시(MD5 키)에 명중됨. */
        const val JPEG_QUALITY = 70

        /** SharedPreferences 키 */
        const val PREF_NAME = "safestep_prefs"
        const val PREF_SERVER_URL = "server_url"

        /** 이 기기의 고유 클라이언트 ID (서버 슬롯 추적용) */
        val CLIENT_ID: String = java.util.UUID.randomUUID().toString()
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(3, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.SECONDS)
        .writeTimeout(5, TimeUnit.SECONDS)
        .build()

    /** 서버가 보낸 마지막 TTS 메시지 (Ollama ON이면 자연어, OFF면 규칙 기반) */
    var lastMessage: String = ""

    /** 서버가 보낸 회피 방향 (왼쪽/정면/오른쪽) - 진동 방향에 사용 */
    var lastDodge: String = "정면"

    /** 마지막 요청의 서버 연결 성공 여부 */
    @Volatile var isConnected: Boolean = true
        private set

    /** 서버가 만석(429)으로 거절했는지 — 클라이언트가 별도 안내용으로 사용 */
    @Volatile var isServerBusy: Boolean = false
        private set

    /** 서버가 부하로 Depth를 자동 OFF했는지 (true면 거리 정보 없음) */
    @Volatile var depthAutoOff: Boolean = false
        private set

    override fun detect(bitmap: Bitmap, rotationDegrees: Int): List<Detection> {
        // 회전 + JPEG 압축 후 detectJpeg에 위임. 파생 비트맵 즉시 recycle (누수 방지)
        val corrected = if (rotationDegrees != 0) {
            val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else bitmap
        val stream = ByteArrayOutputStream()
        corrected.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, stream)
        if (corrected !== bitmap) corrected.recycle()
        return detectJpeg(stream.toByteArray())
    }

    /**
     * 이미 회전+JPEG 인코드된 바이트로 탐지.
     * analyzeFrame에서 한 번만 압축해 detect/segment에 공유할 때 사용.
     * (서버 추론 캐시도 같은 바이트면 명중되어 추론 1회 절감)
     */
    fun detectJpeg(jpegBytes: ByteArray): List<Detection> {
        return try {
            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "file", "frame.jpg",
                    jpegBytes.toRequestBody("image/jpeg".toMediaType())
                )
                .build()

            val request = Request.Builder()
                .url(SERVER_URL)
                .header("X-Client-Id", CLIENT_ID)
                .post(requestBody)
                .build()

            val response = client.newCall(request).execute()
            if (response.code == 429) {
                Log.w(TAG, "서버 만석 (429)")
                isServerBusy = true
                isConnected  = true   // 네트워크는 정상
                return emptyList()
            }
            if (!response.isSuccessful) {
                Log.w(TAG, "서버 응답 실패: ${response.code}")
                return emptyList()
            }

            val body = response.body?.string() ?: return emptyList()
            val root = JSONObject(body)
            lastMessage = root.optString("message", "")
            lastDodge   = root.optString("dodge", "정면")
            isConnected  = true
            isServerBusy = false
            // Depth 자동 OFF 상태 추출
            depthAutoOff = root.optJSONObject("depth_status")?.optBoolean("auto_off", false) ?: false
            parseResponse(body)
        } catch (e: Exception) {
            Log.w(TAG, "서버 통신 실패: ${e.message}")
            isConnected = false
            emptyList()
        }
    }

    private fun parseResponse(json: String): List<Detection> {
        val result = mutableListOf<Detection>()
        try {
            val root = JSONObject(json)
            val arr = root.getJSONArray("detections")
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                // label_ko = 서버가 보내는 영문 그룹명 → 한국어로 변환
                val groupKo = when (obj.getString("label_ko")) {
                    "vehicle" -> "차량"
                    "micro"   -> "개인이동장치"
                    "person"  -> "사람/동물"
                    "fixed"   -> "고정장애물"
                    "signal"  -> "신호등/표지판"
                    else      -> obj.getString("label_ko")
                }
                val confidence = obj.getDouble("confidence").toFloat()
                val box = obj.getJSONArray("box")
                val rect = RectF(
                    box.getDouble(0).toFloat(),  // x1
                    box.getDouble(1).toFloat(),  // y1
                    box.getDouble(2).toFloat(),  // x2
                    box.getDouble(3).toFloat()   // y2
                )
                // 서버가 보내는 방향 및 거리 파싱
                val direction = obj.optString("direction", "정면")
                val depthM = if (obj.has("depth_m") && !obj.isNull("depth_m"))
                    obj.getDouble("depth_m").toFloat() else null

                result.add(Detection(groupKo, confidence, rect, direction, depthM))
            }
        } catch (e: Exception) {
            Log.e(TAG, "JSON 파싱 실패: ${e.message}")
        }
        return result
    }

    override fun close() {
        client.dispatcher.executorService.shutdown()
        client.connectionPool.evictAll()
    }
}
