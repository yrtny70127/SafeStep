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

        // ★ 앱 실행 시 SplashActivity가 SharedPreferences → 이 변수에 설정
        //   직접 수정 시 이 값을 교체하면 됩니다
        var SERVER_URL = "https://ungloved-grill-unlovely.ngrok-free.dev/detect"

        // /fast 엔드포인트 URL (SERVER_URL에서 자동 파생)
        val FAST_URL get() = SERVER_URL
            .substringBeforeLast("/detect")
            .substringBeforeLast("/fast") + "/fast"

        /** 일반 탐지 — 전송 JPEG 품질 / 최대 해상도 */
        private const val JPEG_QUALITY = 35
        private const val MAX_SIDE     = 288

        /** 고속 탐지 — 더 작은 이미지로 응답 속도 극대화 */
        private const val FAST_JPEG_QUALITY = 20
        private const val FAST_MAX_SIDE     = 160
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(3, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.SECONDS)
        .writeTimeout(5, TimeUnit.SECONDS)
        .build()

    override fun detect(bitmap: Bitmap, rotationDegrees: Int): List<Detection> {
        return try {
            // 1) 회전 보정
            val corrected = if (rotationDegrees != 0) {
                val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            } else bitmap

            // 2) 리사이즈 — 긴 변을 MAX_SIDE 이하로 축소 (전송 속도 향상)
            val scale = MAX_SIDE.toFloat() / maxOf(corrected.width, corrected.height)
            val toSend = if (scale < 1f)
                Bitmap.createScaledBitmap(
                    corrected,
                    (corrected.width  * scale).toInt(),
                    (corrected.height * scale).toInt(),
                    false
                )
            else corrected

            // 3) Bitmap → JPEG 바이트
            val stream = ByteArrayOutputStream()
            toSend.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, stream)
            val jpegBytes = stream.toByteArray()

            // 2) Multipart POST 전송
            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "file", "frame.jpg",
                    jpegBytes.toRequestBody("image/jpeg".toMediaType())
                )
                .build()

            val request = Request.Builder()
                .url(SERVER_URL)
                .post(requestBody)
                .build()

            val response = client.newCall(request).execute()
            if (!response.isSuccessful) {
                Log.w(TAG, "서버 응답 실패: ${response.code}")
                return emptyList()
            }

            // 3) JSON 파싱 → List<Detection>
            val body = response.body?.string() ?: return emptyList()
            parseResponse(body)

        } catch (e: Exception) {
            Log.w(TAG, "서버 통신 실패: ${e.message}")
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
                val label = obj.getString("label")
                val labelKo = obj.getString("label_ko")
                val confidence = obj.getDouble("confidence").toFloat()
                val box = obj.getJSONArray("box")
                val rect = RectF(
                    box.getDouble(0).toFloat(),  // x1
                    box.getDouble(1).toFloat(),  // y1
                    box.getDouble(2).toFloat(),  // x2
                    box.getDouble(3).toFloat()   // y2
                )
                // depth_m: null 이면 서버 depth 모델 미로드 상태
                val depthM = if (obj.isNull("depth_m")) null
                             else obj.optDouble("depth_m", 0.0).toFloat()
                // group / approach_speed (접근 속도 기반 경고용)
                val group = obj.optString("group").takeIf { it.isNotEmpty() && it != "null" }
                val approachSpeed = if (obj.isNull("approach_speed")) null
                                    else obj.optDouble("approach_speed", 0.0).toFloat()
                // label_ko 를 label 필드에 담아서 앱에서 바로 출력
                result.add(Detection(labelKo, confidence, rect, depthM, group, approachSpeed))
            }
        } catch (e: Exception) {
            Log.e(TAG, "JSON 파싱 실패: ${e.message}")
        }
        return result
    }

    /**
     * 고속 경량 탐지 — /fast 엔드포인트 (imgsz=160, 차량·사람만).
     * 매 프레임 호출해 충돌 경고 반응 속도를 극대화.
     * rotationDegrees 는 analyzeFrame() 에서 이미 보정된 값 전달 시 0 으로 설정.
     */
    fun detectFast(bitmap: Bitmap, rotationDegrees: Int): List<Detection> {
        return try {
            val corrected = if (rotationDegrees != 0) {
                val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            } else bitmap

            val scale = FAST_MAX_SIDE.toFloat() / maxOf(corrected.width, corrected.height)
            val toSend = if (scale < 1f)
                Bitmap.createScaledBitmap(
                    corrected,
                    (corrected.width  * scale).toInt(),
                    (corrected.height * scale).toInt(),
                    false
                )
            else corrected

            val stream = ByteArrayOutputStream()
            toSend.compress(Bitmap.CompressFormat.JPEG, FAST_JPEG_QUALITY, stream)
            val jpegBytes = stream.toByteArray()

            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", "frame.jpg",
                    jpegBytes.toRequestBody("image/jpeg".toMediaType()))
                .build()

            val request = Request.Builder().url(FAST_URL).post(requestBody).build()
            val response = client.newCall(request).execute()
            if (!response.isSuccessful) return emptyList()

            parseResponse(response.body?.string() ?: return emptyList())
        } catch (e: Exception) {
            Log.w(TAG, "고속 탐지 실패: ${e.message}")
            emptyList()
        }
    }

    override fun close() {
        client.dispatcher.executorService.shutdown()
        client.connectionPool.evictAll()
    }
}
