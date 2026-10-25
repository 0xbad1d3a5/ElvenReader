package com.androidreader.ui

import android.Manifest
import android.content.ComponentName
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import androidx.media3.session.MediaController
import androidx.media3.session.SessionToken
import com.androidreader.drama.Dramatization
import com.androidreader.net.ServerSettings
import com.androidreader.net.normalizeServerUrl
import com.androidreader.player.PlaybackHub
import com.androidreader.player.PlaybackService
import com.google.common.util.concurrent.ListenableFuture

sealed class Screen {
    data object Library : Screen()
    data class Reader(val bookId: String) : Screen()
    data object Settings : Screen()
    data class Pronunciations(val bookId: String?) : Screen()
    data class Cast(val bookId: String) : Screen()
}

class MainActivity : ComponentActivity() {
    private val pairing = mutableStateOf<Pair<List<String>, String>?>(null)
    private val stack = mutableStateListOf<Screen>(Screen.Library)
    private var controller: ListenableFuture<MediaController>? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        PlaybackHub.init(this)
        Dramatization.init(this)
        if (!ServerSettings(this).isConfigured) stack.add(Screen.Settings)
        handleIntent(intent)
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
        }
        setContent { AppTheme { Root() } }
    }

    override fun onStart() {
        super.onStart()
        // Connecting a controller starts the playback service (and so the player).
        val token = SessionToken(this, ComponentName(this, PlaybackService::class.java))
        controller = MediaController.Builder(this, token).buildAsync()
    }

    override fun onStop() {
        controller?.let { MediaController.releaseFuture(it) }
        controller = null
        super.onStop()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(intent: Intent?) {
        if (intent?.getStringExtra(EXTRA_SCREEN) == "settings" && stack.lastOrNull() != Screen.Settings) {
            stack.add(Screen.Settings)
        }
        val data = intent?.data ?: return
        if (data.scheme != "areader" || data.host != "pair") return
        val urls = data.getQueryParameters("url").map { normalizeServerUrl(it) }.filter { it.isNotEmpty() }
        val token = data.getQueryParameter("token")
        if (urls.isNotEmpty() && !token.isNullOrBlank()) {
            pairing.value = urls to token
            if (stack.lastOrNull() != Screen.Settings) stack.add(Screen.Settings)
        }
    }

    @Composable
    private fun Root() {
        val screen = stack.last()
        BackHandler(enabled = stack.size > 1) { stack.removeAt(stack.size - 1) }
        val back = { if (stack.size > 1) stack.removeAt(stack.size - 1) else finish() }
        Box(Modifier.fillMaxSize()) {
            when (screen) {
                Screen.Library -> LibraryScreen(
                    onOpen = { stack.add(Screen.Reader(it)) },
                    onSettings = { stack.add(Screen.Settings) },
                    onPronunciations = { stack.add(Screen.Pronunciations(null)) },
                )
                is Screen.Reader -> ReaderScreen(
                    bookId = screen.bookId,
                    onBack = { back() },
                    onPronunciations = { stack.add(Screen.Pronunciations(screen.bookId)) },
                    onCast = { stack.add(Screen.Cast(screen.bookId)) },
                )
                Screen.Settings -> SettingsScreen(pairing, onBack = { back() })
                is Screen.Pronunciations -> PronunciationsScreen(screen.bookId, onBack = { back() })
                is Screen.Cast -> CastScreen(screen.bookId, onBack = { back() })
            }
        }
    }

    companion object {
        const val EXTRA_SCREEN = "screen"
    }
}

/** Target of the gear icon in the system TTS settings. */
class EngineSettingsActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        startActivity(Intent(this, MainActivity::class.java).putExtra(MainActivity.EXTRA_SCREEN, "settings"))
        finish()
    }
}
