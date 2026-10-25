package com.androidreader.ui

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.selection.selectable
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import com.androidreader.player.SpeechAudio

@Composable
fun AppTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val context = LocalContext.current
    val colors = when {
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S ->
            if (dark) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        dark -> darkColorScheme()
        else -> lightColorScheme()
    }
    MaterialTheme(colorScheme = colors) {
        Surface(modifier = Modifier.fillMaxSize(), content = content)
    }
}

/** Page colours for the reader: background, text, highlight. */
data class PageColors(val background: Color, val text: Color, val highlight: Color, val muted: Color)

@Composable
fun pageColors(theme: String): PageColors {
    val dark = when (theme) {
        "dark" -> true
        "light", "sepia" -> false
        else -> isSystemInDarkTheme()
    }
    return when {
        theme == "sepia" -> PageColors(Color(0xFFF4ECD8), Color(0xFF3B2F1E), Color(0xFFE5C984), Color(0xFF7A6A50))
        dark -> PageColors(Color(0xFF121212), Color(0xFFE2E2E2), Color(0xFF3D4F7A), Color(0xFF9A9A9A))
        else -> PageColors(Color(0xFFFFFFFF), Color(0xFF1C1C1C), Color(0xFFFFE69A), Color(0xFF6B6B6B))
    }
}

@Composable
fun Section(title: String, content: @Composable () -> Unit) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            content()
        }
    }
}

@Composable
fun ChoiceRow(selected: Boolean, label: String, onClick: () -> Unit) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .selectable(selected = selected, onClick = onClick, role = Role.RadioButton),
    ) {
        RadioButton(selected = selected, onClick = null)
        Text(label, modifier = Modifier.padding(start = 8.dp))
    }
}

/** Plays a short clip (pronunciation previews) outside the reading session. */
object ClipPlayer {
    private var track: AudioTrack? = null

    @Synchronized
    fun play(audio: SpeechAudio) {
        stop()
        if (audio.pcm.isEmpty()) return
        val t = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build(),
            )
            .setAudioFormat(
                AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(audio.sampleRate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build(),
            )
            .setTransferMode(AudioTrack.MODE_STATIC)
            .setBufferSizeInBytes(audio.pcm.size)
            .build()
        t.write(audio.pcm, 0, audio.pcm.size)
        t.play()
        track = t
    }

    @Synchronized
    fun stop() {
        track?.run {
            runCatching { stop() }
            release()
        }
        track = null
    }
}
