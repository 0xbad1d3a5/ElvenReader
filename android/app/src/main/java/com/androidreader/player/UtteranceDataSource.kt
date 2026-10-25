package com.androidreader.player

import android.net.Uri
import androidx.media3.common.C
import androidx.media3.common.util.UnstableApi
import androidx.media3.datasource.BaseDataSource
import androidx.media3.datasource.DataSource
import androidx.media3.datasource.DataSpec
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Serves one utterance as a WAV file to ExoPlayer. The URI only identifies the
 * utterance (`areader://u?c=<chapter>&i=<index>&g=<generation>`); the audio comes
 * from [PlaybackHub], which fetches it from the PC (usually well in advance).
 * While the PC is still synthesizing, open() blocks and the player shows
 * buffering.
 */
@androidx.annotation.OptIn(UnstableApi::class)
class UtteranceDataSource : BaseDataSource(/* isNetwork = */ true) {
    private var uri: Uri? = null
    private var opened = false
    private var data: ByteArray = ByteArray(0)
    private var position = 0
    private var remaining = 0

    override fun open(dataSpec: DataSpec): Long {
        uri = dataSpec.uri
        transferInitializing(dataSpec)
        val audio = PlaybackHub.awaitAudio(dataSpec.uri)
        data = wav(audio)
        if (dataSpec.position > data.size) throw IOException("position beyond end")
        position = dataSpec.position.toInt()
        remaining = data.size - position
        if (dataSpec.length != C.LENGTH_UNSET.toLong()) remaining = minOf(remaining, dataSpec.length.toInt())
        opened = true
        transferStarted(dataSpec)
        return remaining.toLong()
    }

    override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
        if (length == 0) return 0
        if (remaining == 0) return C.RESULT_END_OF_INPUT
        val n = minOf(length, remaining)
        System.arraycopy(data, position, buffer, offset, n)
        position += n
        remaining -= n
        bytesTransferred(n)
        return n
    }

    override fun getUri(): Uri? = uri

    override fun close() {
        uri = null
        if (opened) {
            opened = false
            transferEnded()
        }
    }

    class Factory : DataSource.Factory {
        override fun createDataSource(): DataSource = UtteranceDataSource()
    }

    companion object {
        fun wav(audio: SpeechAudio): ByteArray {
            val pcm = if (audio.pcm.isEmpty()) ByteArray(audio.sampleRate / 10 * 2) else audio.pcm
            val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            header.put("RIFF".toByteArray()).putInt(36 + pcm.size).put("WAVE".toByteArray())
                .put("fmt ".toByteArray()).putInt(16).putShort(1).putShort(1)
                .putInt(audio.sampleRate).putInt(audio.sampleRate * 2).putShort(2).putShort(16)
                .put("data".toByteArray()).putInt(pcm.size)
            return header.array() + pcm
        }
    }
}
