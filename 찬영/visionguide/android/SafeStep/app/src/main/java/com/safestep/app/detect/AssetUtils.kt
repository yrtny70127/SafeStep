package com.safestep.app.detect

import android.content.Context
import java.io.File

/**
 * PyTorch Mobile 의 Module.load() 는 실제 파일 경로를 요구한다.
 * APK 안의 assets 는 파일 경로가 없으므로, cacheDir 로 한 번 복사한 뒤 그 경로를 넘겨야 한다.
 *
 * 이미 같은 크기 파일이 캐시에 있으면 재사용한다 (앱 업데이트로 모델이 바뀌면 새로 복사).
 */
object AssetUtils {

    fun assetFilePath(context: Context, assetName: String): String? {
        val assets = runCatching { context.assets.list("")?.toList().orEmpty() }
            .getOrDefault(emptyList())
        if (assetName !in assets) return null

        val outFile = File(context.filesDir, assetName)
        // 크기 비교로 변경 여부 체크 (간단)
        val assetSize = context.assets.openFd(assetName).use { it.length }
        if (outFile.exists() && outFile.length() == assetSize) return outFile.absolutePath

        context.assets.open(assetName).use { input ->
            outFile.outputStream().use { output -> input.copyTo(output) }
        }
        return outFile.absolutePath
    }

    fun readAssetLines(context: Context, assetName: String): List<String> =
        runCatching {
            context.assets.open(assetName).bufferedReader().useLines { it.toList() }
                .map { it.trim() }
                .filter { it.isNotEmpty() }
        }.getOrDefault(emptyList())
}
