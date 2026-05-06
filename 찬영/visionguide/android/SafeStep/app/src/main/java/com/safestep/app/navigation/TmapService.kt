package com.safestep.app.navigation

import android.os.Handler
import android.os.Looper
import android.util.Log
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import org.osmdroid.util.GeoPoint
import java.util.concurrent.TimeUnit

// ── 데이터 클래스 ──────────────────────────────────────────────────────────────

data class PoiResult(
    val name: String,
    val lat: Double,
    val lon: Double
)

data class RouteStep(
    val description: String,
    val distance: Int,   // 다음 스텝까지 거리 (m)
    val lat: Double,
    val lon: Double,
    val turnType: Int    // 200=출발, 201=도착, 11=직진, 12=좌회전, 13=우회전 등
)

data class RouteResult(
    val steps: List<RouteStep>,
    val pathPoints: List<GeoPoint>,
    val totalDistanceM: Int,
    val totalTimeSec: Int
)

// ── Tmap REST API ──────────────────────────────────────────────────────────────

object TmapService {

    private const val TAG = "TmapService"

    // ★ Tmap API 키 (local.properties → BuildConfig.TMAP_API_KEY)
    val API_KEY: String get() = com.safestep.app.BuildConfig.TMAP_API_KEY
    private const val BASE = "https://apis.openapi.sk.com/tmap"

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .build()

    private val main = Handler(Looper.getMainLooper())

    // ── POI 검색 (키워드 → 장소 목록) ──────────────────────────────────────────

    fun searchPoi(keyword: String, onResult: (List<PoiResult>) -> Unit) {
        Thread {
            try {
                val encoded = java.net.URLEncoder.encode(keyword, "UTF-8")
                val url = "$BASE/pois?version=1&searchKeyword=$encoded" +
                        "&resCoordType=WGS84GEO&reqCoordType=WGS84GEO&count=5&appKey=$API_KEY"
                val req = Request.Builder().url(url).get().build()
                val body = client.newCall(req).execute().body?.string() ?: ""
                val result = parsePoi(body)
                main.post { onResult(result) }
            } catch (e: Exception) {
                Log.e(TAG, "POI 검색 실패", e)
                main.post { onResult(emptyList()) }
            }
        }.start()
    }

    private fun parsePoi(json: String): List<PoiResult> {
        val list = mutableListOf<PoiResult>()
        return try {
            val pois = JSONObject(json)
                .getJSONObject("searchPoiInfo")
                .getJSONObject("pois")
                .getJSONArray("poi")
            for (i in 0 until minOf(pois.length(), 5)) {
                val p = pois.getJSONObject(i)
                list.add(PoiResult(
                    name = p.getString("name"),
                    lat  = p.getString("noorLat").toDouble(),
                    lon  = p.getString("noorLon").toDouble()
                ))
            }
            list
        } catch (e: Exception) {
            Log.e(TAG, "POI 파싱 실패: $e")
            list
        }
    }

    // ── 가까운 정류장 검색 (대중교통 안내용) ─────────────────────────────────
    // T맵 POI 검색은 키워드 + 중심 좌표 + 반경 지원. 가장 가까운 1개를 돌려준다.
    fun searchNearestTransit(
        centerLat: Double, centerLon: Double,
        keyword: String = "버스정류장",
        radiusM: Int = 500,
        onResult: (PoiResult?) -> Unit
    ) {
        Thread {
            try {
                val encoded = java.net.URLEncoder.encode(keyword, "UTF-8")
                val url = "$BASE/pois?version=1&searchKeyword=$encoded" +
                        "&centerLat=$centerLat&centerLon=$centerLon&radius=${radiusM / 1000}" +
                        "&resCoordType=WGS84GEO&reqCoordType=WGS84GEO&count=10&appKey=$API_KEY"
                val req = Request.Builder().url(url).get().build()
                val body = client.newCall(req).execute().body?.string() ?: ""
                val pois = parsePoi(body)
                // 가장 가까운 1개 (해버사인 거리)
                val nearest = pois.minByOrNull { distM(centerLat, centerLon, it.lat, it.lon) }
                main.post { onResult(nearest) }
            } catch (e: Exception) {
                Log.e(TAG, "정류장 검색 실패", e)
                main.post { onResult(null) }
            }
        }.start()
    }

    private fun distM(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val r = 6_371_000.0
        val dLat = Math.toRadians(lat2 - lat1)
        val dLon = Math.toRadians(lon2 - lon1)
        val a = Math.sin(dLat / 2).let { it * it } +
                Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) *
                Math.sin(dLon / 2).let { it * it }
        return 2 * r * Math.asin(Math.sqrt(a))
    }

    // ── 도보 경로 탐색 ──────────────────────────────────────────────────────────

    fun searchPedestrianRoute(
        startLat: Double, startLon: Double,
        endLat: Double,   endLon: Double,
        endName: String,
        onResult: (RouteResult?) -> Unit
    ) {
        Thread {
            try {
                val bodyJson = JSONObject().apply {
                    put("startX",       startLon.toString())
                    put("startY",       startLat.toString())
                    put("endX",         endLon.toString())
                    put("endY",         endLat.toString())
                    put("startName",    "출발지")
                    put("endName",      endName)
                    put("reqCoordType", "WGS84GEO")
                    put("resCoordType", "WGS84GEO")
                    put("searchOption", "0")
                }.toString()

                val req = Request.Builder()
                    .url("$BASE/routes/pedestrian?version=1")
                    .post(bodyJson.toRequestBody("application/json".toMediaType()))
                    .addHeader("appKey", API_KEY)
                    .build()

                val body = client.newCall(req).execute().body?.string() ?: ""
                val result = parseRoute(body)
                main.post { onResult(result) }
            } catch (e: Exception) {
                Log.e(TAG, "경로 탐색 실패", e)
                main.post { onResult(null) }
            }
        }.start()
    }

    private fun parseRoute(json: String): RouteResult? {
        return try {
            val features = JSONObject(json).getJSONArray("features")
            val steps      = mutableListOf<RouteStep>()
            val pathPoints = mutableListOf<GeoPoint>()
            var totalDist  = 0
            var totalTime  = 0

            for (i in 0 until features.length()) {
                val feat  = features.getJSONObject(i)
                val geo   = feat.getJSONObject("geometry")
                val props = feat.getJSONObject("properties")

                when (geo.getString("type")) {
                    "Point" -> {
                        val c = geo.getJSONArray("coordinates")
                        // 첫 포인트에서 전체 거리/시간 추출
                        if (steps.isEmpty()) {
                            totalDist = props.optInt("totalDistance", 0)
                            totalTime = props.optInt("totalTime", 0)
                        }
                        steps.add(RouteStep(
                            description = props.optString("description", ""),
                            distance    = props.optInt("distance", 0),
                            lat         = c.getDouble(1),
                            lon         = c.getDouble(0),
                            turnType    = props.optInt("turnType", 0)
                        ))
                    }
                    "LineString" -> {
                        val coords = geo.getJSONArray("coordinates")
                        for (j in 0 until coords.length()) {
                            val c = coords.getJSONArray(j)
                            pathPoints.add(GeoPoint(c.getDouble(1), c.getDouble(0)))
                        }
                    }
                }
            }
            RouteResult(steps, pathPoints, totalDist, totalTime)
        } catch (e: Exception) {
            Log.e(TAG, "경로 파싱 실패: $e")
            null
        }
    }
}
