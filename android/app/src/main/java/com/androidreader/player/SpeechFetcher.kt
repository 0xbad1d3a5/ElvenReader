package com.androidreader.player

import android.content.Context
import com.androidreader.net.ServerClient
import com.androidreader.net.TtsRequest
import okhttp3.Call
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.InterruptedIOException
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutionException
import java.util.concurrent.LinkedBlockingDeque
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

class SpeechAudio(val sampleRate: Int, val pcm: ByteArray) {
    val durationMs: Long get() = pcm.size / 2 * 1000L / sampleRate.coerceAtLeast(1)
}

/**
 * Fetches synthesized utterances from the PC, one at a time, ahead of playback.
 *
 * Quality models can be slower than real time on long sentences, so the reader
 * asks for several utterances ahead; each is fetched in full and kept in memory
 * until played. The PC is a single GPU, so requests are strictly sequential; an
 * urgent request (the one playback is waiting for) jumps the queue, and jumps
 * elsewhere in the book cancel work that is no longer needed.
 */
class SpeechFetcher(context: Context) {
    private val client = ServerClient.get(context)
    private val lock = Object()
    private val futures = LinkedHashMap<String, CompletableFuture<SpeechAudio>>()
    private val requests = HashMap<String, TtsRequest>()
    private val queue = LinkedBlockingDeque<String>()
    @Volatile private var running: Pair<String, Call?>? = null

    /** Called with (key, ok) when a fetch finishes, for buffering indicators. */
    @Volatile var listener: ((String, Throwable?) -> Unit)? = null

    init {
        Thread(::loop, "speech-fetcher").apply { isDaemon = true }.start()
    }

    fun request(key: String, req: TtsRequest, urgent: Boolean = false): CompletableFuture<SpeechAudio> = synchronized(lock) {
        futures[key]?.let { existing ->
            if (urgent && !existing.isDone && running?.first != key) {
                queue.remove(key)
                queue.addFirst(key)
            }
            return existing
        }
        val f = CompletableFuture<SpeechAudio>()
        futures[key] = f
        requests[key] = req
        if (urgent) queue.addFirst(key) else queue.addLast(key)
        trim()
        f
    }

    fun peek(key: String): CompletableFuture<SpeechAudio>? = synchronized(lock) { futures[key] }

    fun isReady(key: String): Boolean = synchronized(lock) { futures[key]?.let { it.isDone && !it.isCompletedExceptionally } == true }

    /** Blocks until the audio for [key] is available; interruptible. */
    fun await(key: String, req: TtsRequest, timeoutMs: Long): SpeechAudio {
        val f = request(key, req, urgent = true)
        val deadline = System.currentTimeMillis() + timeoutMs
        while (true) {
            if (Thread.interrupted()) throw InterruptedIOException("cancelled")
            try {
                return f.get(250, TimeUnit.MILLISECONDS)
            } catch (_: TimeoutException) {
                if (System.currentTimeMillis() > deadline) throw IOException("The PC took too long to synthesize this sentence")
            } catch (e: InterruptedException) {
                throw InterruptedIOException("cancelled")
            } catch (e: ExecutionException) {
                synchronized(lock) { if (futures[key] === f) forget(key) }
                val cause = e.cause
                throw if (cause is IOException) cause else IOException(cause?.message ?: "synthesis failed", cause)
            }
        }
    }

    /** Keeps only [keep]; everything else is dropped and an in-flight request is cancelled. */
    fun retainOnly(keep: Set<String>) = synchronized(lock) {
        for (k in futures.keys.toList()) {
            if (k !in keep) forget(k)
        }
        running?.let { (k, call) -> if (k !in keep) call?.cancel() }
    }

    fun clear() = retainOnly(emptySet())

    private fun forget(key: String) {
        queue.remove(key)
        requests.remove(key)
        futures.remove(key)?.let { if (!it.isDone) it.completeExceptionally(InterruptedIOException("dropped")) }
    }

    private fun trim() {
        // Bound memory: ~60 finished utterances (about 1-2 MB each at most).
        val done = futures.filter { it.value.isDone }.keys
        val excess = done.size - 60
        if (excess > 0) done.take(excess).forEach { forget(it) }
    }

    private fun loop() {
        while (true) {
            val key = queue.take()
            val (req, future) = synchronized(lock) {
                val f = futures[key] ?: return@synchronized null to null
                if (f.isDone) return@synchronized null to null
                running = key to null
                requests[key] to f
            }
            if (req == null || future == null) continue
            try {
                val audio = fetch(key, req)
                future.complete(audio)
                listener?.invoke(key, null)
            } catch (e: Throwable) {
                future.completeExceptionally(e)
                synchronized(lock) { if (futures[key] === future) futures.remove(key); requests.remove(key) }
                listener?.invoke(key, e)
            } finally {
                running = null
            }
        }
    }

    private fun fetch(key: String, req: TtsRequest): SpeechAudio {
        client.tts(req) { call ->
            synchronized(lock) {
                if (futures[key] == null) call.cancel()
                running = key to call
            }
        }.use { resp ->
            val rate = resp.header("X-Sample-Rate")?.toIntOrNull() ?: 24000
            val out = ByteArrayOutputStream()
            resp.body?.byteStream()?.copyTo(out) ?: throw IOException("empty response")
            return SpeechAudio(rate, out.toByteArray())
        }
    }
}
