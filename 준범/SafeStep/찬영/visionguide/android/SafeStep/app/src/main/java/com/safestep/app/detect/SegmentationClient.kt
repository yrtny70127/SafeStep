package com.safestep.app.detect

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

data class SegmentResult(
    val roadRatio: Float,          // 중앙 구역 차도 비율 0~1
    val sidewalkRatio: Float,      // 중앙 구역 보도 비율 0~1
    val status: String,            // "sidewalk" | "road" | "crosswalk" | "alley" | "unknown"
    val frontCls: String = "",     // 정면 노면 세부 클래스명 (예: "caution_zone_manhole")
    val leftStatus: String = "",   // 왼쪽 구역 status (방향 유도용)
    val rightStatus: String = "",  // 오른쪽 구역 status (방향 유도용)
    val trafficLightColor: String = ""  // "red" | "green" | "" (횡단보도일 때만)
)

class SegmentationClient(serverUrl: String) {

    companion object {
        private const val TAG = "SegmentClient"
        // RemoteDetector.JPEG_QUALITY와 동일해야 서버측 추론 캐시(MD5 키)에 명중됨
        private const val JPEG_QUALITY = RemoteDetector.JPEG_QUALITY
    }

    private val segmentUrl = serverUrl.trimEnd('/').let {
        it.substringBeforeLast("/detect").substringBeforeLast("/segment") + "/segment"
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .writeTimeout(5, TimeUnit.SECONDS)
        .build()

    /** Activity onDestroy 등에서 호출 — OkHttp 커넥션풀·디스패처 thread 정리 */
    fun close() {
        runCatching {
            client.dispatcher.executorService.shutdown()
            client.connectionPool.evictAll()
        }
    }

    /** 동기 호출 — 반드시 백그라운드 스레드에서 실행 */
    fun segment(bitmap: Bitmap, rotationDegrees: Int): SegmentResult? {
        val corrected = if (rotationDegrees != 0) {
            val m = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
        } else bitmap
        val stream = ByteArrayOutputStream()
        corrected.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, stream)
        if (corrected !== bitmap) corrected.recycle()   // 파생 비트맵 정리
        return segmentJpeg(stream.toByteArray())
    }

    /**
     * 이미 회전+JPEG 인코드된 바이트로 세그멘테이션.
     * detect와 같은 바이트를 보내면 서버 캐시 명중되어 추론 1회만 실행.
     */
    fun segmentJpeg(jpegBytes: ByteArray): SegmentResult? {
        return try {
            val body = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", "frame.jpg",
                    jpegBytes.toRequestBody("image/jpeg".toMediaType()))
                .build()

            val req = Request.Builder()
                .url(segmentUrl)
                .header("X-Client-Id", RemoteDetector.CLIENT_ID)
                .post(body)
                .build()
            val resp = client.newCall(req).execute()

            if (resp.code == 429) return null   // 만석 (RemoteDetector에서 안내)
            if (!resp.isSuccessful) {
                Log.w(TAG, "서버 응답 실패: ${resp.code}")
                return null
            }

            parseResponse(resp.body?.string() ?: return null)
        } catch (e: Exception) {
            Log.w(TAG, "세그멘테이션 통신 실패: ${e.message}")
            null
        }
    }

    private fun parseResponse(json: String): SegmentResult? {
        return try {
            val obj    = JSONObject(json)
            val status = obj.getString("status")

            val ratios        = obj.optJSONObject("ratios")
            val roadRatio     = ratios?.optDouble("road",     0.0)?.toFloat()
                ?: obj.optDouble("road_ratio",     0.0).toFloat()
            val sidewalkRatio = ratios?.optDouble("sidewalk", 0.0)?.toFloat()
                ?: obj.optDouble("sidewalk_ratio", 0.0).toFloat()

            SegmentResult(
                roadRatio         = roadRatio,
                sidewalkRatio     = sidewalkRatio,
                status            = status,
                frontCls          = obj.optString("front_cls",           ""),
                leftStatus        = obj.optString("left_status",         ""),
                rightStatus       = obj.optString("right_status",        ""),
                trafficLightColor = obj.optString("traffic_light_color", "")
            )
        } catch (e: Exception) {
            Log.e(TAG, "응답 파싱 실패: ${e.message}")
            null
        }
    }
}
