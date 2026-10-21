package com.androidvoicestudio.tts

/**
 * Re-chunks a PCM16 byte stream for [android.speech.tts.SynthesisCallback.audioAvailable]:
 * every emitted block holds whole samples (even byte count) and is no larger than
 * the callback's maxBufferSize. Data is emitted as soon as it arrives so playback
 * starts with minimal delay; a dangling odd byte is carried over to the next feed.
 *
 * [emit] returns false when the framework rejects audio (e.g. playback was stopped);
 * feeding then stops and [feed] returns false.
 */
class PcmChunker(maxBufferSize: Int, private val emit: (ByteArray, Int, Int) -> Boolean) {
    private val max = maxBufferSize.coerceAtLeast(2) and 1.inv()
    private val buffer = ByteArray(max)
    private var fill = 0

    fun feed(data: ByteArray, length: Int = data.size): Boolean {
        var i = 0
        while (i < length) {
            val n = minOf(length - i, max - fill)
            System.arraycopy(data, i, buffer, fill, n)
            fill += n
            i += n
            if (fill == max) {
                if (!emit(buffer, 0, fill)) return false
                fill = 0
            }
        }
        val even = fill and 1.inv()
        if (even > 0) {
            if (!emit(buffer, 0, even)) return false
            if (fill > even) buffer[0] = buffer[even]
            fill -= even
        }
        return true
    }

    /** Bytes held back because they did not form a whole sample. */
    val pendingBytes: Int get() = fill
}
