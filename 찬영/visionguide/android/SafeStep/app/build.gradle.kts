import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
}

// local.properties에서 API 키 읽기
val localProps = Properties().apply {
    val f = rootProject.file("local.properties")
    if (f.exists()) load(f.inputStream())
}
val tmapApiKey: String = localProps.getProperty("TMAP_API_KEY", "")

android {
    namespace = "com.safestep.app"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "com.safestep.app"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // local.properties → BuildConfig.TMAP_API_KEY
        buildConfigField("String", "TMAP_API_KEY", "\"$tmapApiKey\"")
    }

    buildFeatures {
        buildConfig = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    // CameraX
    implementation("androidx.camera:camera-core:1.3.1")
    implementation("androidx.camera:camera-camera2:1.3.1")
    implementation("androidx.camera:camera-lifecycle:1.3.1")
    implementation("androidx.camera:camera-view:1.3.1")

    // HTTP 클라이언트 (서버 통신용)
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // 지도 (OpenStreetMap, API 키 불필요)
    implementation("org.osmdroid:osmdroid-android:6.1.18")

    // GPS / 위치
    implementation("com.google.android.gms:play-services-location:21.2.0")

    // PyTorch Mobile (Lite Interpreter for .ptl models)
    // pytorch_android 와 pytorch_android_lite 는 같은 org.pytorch.Module 클래스를
    // 중복 정의하므로, lite 만 쓰고 torchvision 의 transitive 의존성을 제거한다.
    implementation("org.pytorch:pytorch_android_lite:2.1.0")
    implementation("org.pytorch:pytorch_android_torchvision:2.1.0") {
        exclude(group = "org.pytorch", module = "pytorch_android")
    }
}