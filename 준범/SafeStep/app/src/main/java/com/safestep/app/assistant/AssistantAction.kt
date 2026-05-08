package com.safestep.app.assistant

/**
 * 서버 /assistant 응답을 표현하는 sealed class.
 *
 * 서버 응답 예:
 *   {"action": "start_navigation", "params": {"destination": "강남역"}, "tts": "..."}
 *
 * → AssistantAction.StartNavigation(destination = "강남역", tts = "...")
 *
 * 화면별 분기:
 *   - SplashActivity → EnterCameraMode / EnterVideoTestMode
 *   - MapActivity    → 그 외 모든 action
 */
sealed class AssistantAction {
    abstract val tts: String

    // ─── SplashActivity 전용 ───
    data class EnterCameraMode(override val tts: String) : AssistantAction()
    data class EnterVideoTestMode(override val tts: String) : AssistantAction()

    // ─── MapActivity 전용 ───
    data class StartNavigation(val destination: String, override val tts: String) : AssistantAction()
    data class CancelNavigation(override val tts: String) : AssistantAction()
    data class Reroute(override val tts: String) : AssistantAction()
    data class CurrentLocation(override val tts: String) : AssistantAction()
    data class RemainingInfo(override val tts: String) : AssistantAction()
    data class DescribeSurroundings(override val tts: String) : AssistantAction()
    data class FindNearbyPoi(val category: String, override val tts: String) : AssistantAction()

    // ─── 공통 (분류 안 됨 / 일반 응답) ───
    data class SpeakOnly(val text: String, override val tts: String = text) : AssistantAction()

    companion object {
        /** 서버 응답 JSON → AssistantAction. */
        fun fromJson(action: String, params: Map<String, Any?>, tts: String): AssistantAction =
            when (action) {
                "enter_camera_mode"     -> EnterCameraMode(tts)
                "enter_video_test_mode" -> EnterVideoTestMode(tts)
                "start_navigation"      -> StartNavigation(
                    destination = params["destination"] as? String ?: "",
                    tts         = tts,
                )
                "cancel_navigation"     -> CancelNavigation(tts)
                "reroute"               -> Reroute(tts)
                "get_current_location"  -> CurrentLocation(tts)
                "get_remaining_info"    -> RemainingInfo(tts)
                "describe_surroundings" -> DescribeSurroundings(tts)
                "find_nearby_poi"       -> FindNearbyPoi(
                    category = params["category"] as? String ?: "",
                    tts      = tts,
                )
                else -> SpeakOnly(tts)   // "speak_only" 또는 알 수 없는 action
            }
    }
}
