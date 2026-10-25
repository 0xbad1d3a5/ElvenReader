package com.androidreader.net

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import okhttp3.Call
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.ConnectException
import java.net.NoRouteToHostException
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

data class ServerVoice(
    val id: String,
    val name: String,
    val language: String?,
    val description: String? = null,
    val source: String = "",
)

data class EngineInfo(
    val id: String,
    val label: String,
    val description: String,
    val installed: Boolean,
    val active: Boolean,
    val delivery: String,
    val voiceDesign: Boolean,
    val licence: String,
    val rtf: Double?,
)

data class PronEntry(
    val id: String,
    val term: String,
    val respelling: String,
    val phonemes: String,
    val caseSensitive: Boolean,
    val book: String?,
) {
    val isGlobal: Boolean get() = book == null
}

data class TtsRequest(
    val text: String,
    val engine: String? = null,
    val voice: String? = null,
    val language: String? = null,
    val speed: Float = 1f,
    val delivery: String? = null,
    val sentencePauseMs: Int = ServerSettings.DEFAULT_SENTENCE_PAUSE_MS,
    val book: String? = null,
    val extraPronunciations: List<PronEntry> = emptyList(),
    val cache: Boolean = true,
) {
    fun toJson(): JSONObject {
        val json = JSONObject()
            .put("text", text)
            .put("speed", speed.toDouble())
            .put("sentence_pause_ms", sentencePauseMs)
            .put("cache", cache)
        engine?.takeIf { it.isNotBlank() }?.let { json.put("engine", it) }
        voice?.takeIf { it.isNotBlank() }?.let { json.put("voice", it) }
        language?.takeIf { it.isNotBlank() }?.let { json.put("language", it) }
        delivery?.takeIf { it.isNotBlank() }?.let { json.put("delivery", it) }
        book?.takeIf { it.isNotBlank() }?.let { json.put("book", it) }
        if (extraPronunciations.isNotEmpty()) {
            json.put("extra_pronunciations", JSONArray(extraPronunciations.map {
                JSONObject().put("term", it.term).put("respelling", it.respelling)
                    .put("phonemes", it.phonemes).put("case_sensitive", it.caseSensitive)
            }))
        }
        return json
    }
}

/** HTTP error from the server, carrying its status code and `{"error": ...}` detail. */
class ServerHttpException(val code: Int, message: String) : IOException(message)

/** Every configured URL failed to connect. */
class ServerUnreachableException(message: String, cause: Throwable?) : IOException(message, cause)

/**
 * Talks to the PC server. Blocking; call from a background thread.
 *
 * Several URLs can be configured (home LAN, Tailscale...). Requests go to the
 * URL that worked last; if it can't be *connected to*, the next one is tried,
 * so walking out of the house switches to the tunnel without user action.
 * Errors after a connection was made (HTTP errors, slow synthesis) never fail
 * over, because another URL reaches the same PC.
 */
class ServerClient private constructor(private val context: Context) {
    private val settings = ServerSettings(context)
    @Volatile private var preferred: String? = null
    private val prober = Executors.newSingleThreadExecutor()

    val activeUrl: String? get() = preferred ?: settings.urls.firstOrNull()

    private fun ordered(): List<String> {
        val urls = settings.urls
        val p = preferred
        return if (p != null && p in urls) listOf(p) + (urls - p) else urls
    }

    private fun clientFor(base: OkHttpClient, url: String): Pair<OkHttpClient, Boolean> {
        val network = LocalNetworks.networkFor(context, url) ?: return base to false
        return boundClients.getOrPut(network to base) {
            base.newBuilder().socketFactory(network.socketFactory).build()
        } to true
    }

    /** Runs [block] against each URL until one connects. */
    private fun <T> failover(base: OkHttpClient, block: (url: String, http: OkHttpClient) -> T): T {
        val urls = ordered()
        if (urls.isEmpty() || settings.token.isBlank()) throw ServerUnreachableException("Not paired with a PC yet", null)
        var last: IOException? = null
        for (url in urls) {
            try {
                val (http, bound) = clientFor(base, url)
                val result = try {
                    block(url, http)
                } catch (e: IOException) {
                    if (!bound || !isBindRefusal(e)) throw e
                    LocalNetworks.bindingRefused = true
                    block(url, base)
                }
                preferred = url
                return result
            } catch (e: IOException) {
                if (!isConnectFailure(e)) throw e
                last = e
            }
        }
        throw ServerUnreachableException(
            "Could not reach the PC at ${urls.joinToString()} (${last?.message ?: "no route"})", last,
        )
    }

    private fun request(url: String, path: String) = Request.Builder()
        .url(url + path)
        .header("Authorization", "Bearer ${settings.token}")

    private fun call(base: OkHttpClient, path: String, body: RequestBody? = null, method: String? = null): JSONObject =
        failover(base) { url, http ->
            val builder = request(url, path)
            when {
                method != null -> builder.method(method, body)
                body != null -> builder.post(body)
                else -> builder.get()
            }
            http.newCall(builder.build()).execute().use { resp ->
                ensureSuccess(resp)
                val text = resp.body?.string().orEmpty()
                if (text.isBlank()) JSONObject() else JSONObject(text)
            }
        }

    fun getJson(path: String): JSONObject = call(apiHttp, path)
    fun postJson(path: String, json: JSONObject, slow: Boolean = false): JSONObject =
        call(if (slow) slowHttp else apiHttp, path, json.toString().toRequestBody(JSON))
    fun patchJson(path: String, json: JSONObject): JSONObject =
        call(apiHttp, path, json.toString().toRequestBody(JSON), "PATCH")
    fun delete(path: String): JSONObject = call(apiHttp, path, null, "DELETE")

    /**
     * `POST /api/tts`: the response body is raw PCM16LE mono at `X-Sample-Rate`.
     * [register] receives each [Call] before it runs so the caller can cancel it.
     */
    fun tts(req: TtsRequest, register: (Call) -> Unit = {}): Response {
        val body = req.toJson().toString()
        return failover(ttsHttp) { url, http ->
            val call = http.newCall(request(url, "/api/tts").post(body.toRequestBody(JSON)).build())
            register(call)
            val resp = call.execute()
            if (!resp.isSuccessful) resp.use { ensureSuccess(it) }
            resp
        }
    }

    fun health(): JSONObject = getJson("/api/health")

    fun voices(): Pair<List<ServerVoice>, String?> {
        val json = getJson("/api/voices")
        val arr = json.getJSONArray("voices")
        val list = (0 until arr.length()).map { i ->
            val v = arr.getJSONObject(i)
            ServerVoice(
                id = v.optString("id"),
                name = v.optString("name", v.optString("id")),
                language = v.optStringOrNull("language"),
                description = v.optStringOrNull("description"),
                source = v.optString("source"),
            )
        }
        return list to json.optStringOrNull("default")
    }

    fun engines(): Pair<List<EngineInfo>, String?> {
        val json = getJson("/api/engines")
        val arr = json.getJSONArray("engines")
        val list = (0 until arr.length()).map { i ->
            val e = arr.getJSONObject(i)
            val stats = e.optJSONObject("stats")
            EngineInfo(
                id = e.optString("id"),
                label = e.optString("label"),
                description = e.optString("description"),
                installed = e.optBoolean("installed"),
                active = e.optBoolean("active"),
                delivery = e.optString("delivery"),
                voiceDesign = e.optBoolean("voice_design"),
                licence = e.optString("licence"),
                rtf = stats?.optDouble("rtf")?.takeIf { !it.isNaN() },
            )
        }
        return list to json.optStringOrNull("default")
    }

    fun pronunciations(book: String?): List<PronEntry> {
        val json = getJson("/api/pronunciations" + (book?.let { "?book=$it" } ?: ""))
        return parseEntries(json.getJSONArray("global")) + parseEntries(json.getJSONArray("book"))
    }

    fun savePronunciation(entry: PronEntry): PronEntry {
        val body = JSONObject()
            .put("term", entry.term)
            .put("respelling", entry.respelling)
            .put("phonemes", entry.phonemes)
            .put("case_sensitive", entry.caseSensitive)
        val json = if (entry.id.isNotEmpty()) {
            patchJson("/api/pronunciations/${entry.id}", body)
        } else {
            body.put("scope", if (entry.book == null) "global" else "book")
            entry.book?.let { body.put("book", it) }
            postJson("/api/pronunciations", body)
        }
        return parseEntry(json)
    }

    fun deletePronunciation(id: String) {
        delete("/api/pronunciations/$id")
    }

    /** Pings every URL in parallel and prefers the first (in configured order) that answers. */
    fun probe(): String? {
        val urls = settings.urls
        if (urls.isEmpty()) return null
        val results = urls.map { url ->
            Executors.newSingleThreadExecutor().let { ex ->
                ex.submit<Boolean> {
                    try {
                        val (http, _) = clientFor(probeHttp, url)
                        http.newCall(Request.Builder().url("$url/api/ping").build()).execute().use { it.isSuccessful }
                    } catch (_: Exception) {
                        false
                    }
                }.also { ex.shutdown() }
            }
        }
        val ok = urls.zip(results).firstOrNull { (_, f) -> runCatching { f.get(5, TimeUnit.SECONDS) }.getOrDefault(false) }
        ok?.let { preferred = it.first }
        return ok?.first
    }

    private fun watchNetworks() {
        val cm = context.getSystemService(ConnectivityManager::class.java) ?: return
        val request = NetworkRequest.Builder().addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET).build()
        val callback = object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) = reprobe()
            override fun onLost(network: Network) = reprobe()
        }
        try {
            cm.registerNetworkCallback(request, callback)
        } catch (_: Exception) {
        }
    }

    /** A network came or went (home Wi-Fi <-> mobile data): start again from the LAN URL. */
    fun reprobe() {
        preferred = null
        prober.execute { probe() }
    }

    companion object {
        private val JSON = "application/json; charset=utf-8".toMediaType()
        private val boundClients = ConcurrentHashMap<Pair<Network, OkHttpClient>, OkHttpClient>()

        /** Short connect timeout so an unreachable URL fails over quickly. */
        private val baseHttp: OkHttpClient = OkHttpClient.Builder()
            .connectTimeout(4, TimeUnit.SECONDS)
            .writeTimeout(15, TimeUnit.SECONDS)
            .readTimeout(20, TimeUnit.SECONDS)
            .build()
        private val apiHttp = baseHttp
        private val probeHttp = baseHttp.newBuilder().connectTimeout(3, TimeUnit.SECONDS).readTimeout(4, TimeUnit.SECONDS).build()

        /** Synthesis may wait for a quality model to load (the server gives up first and says why). */
        val ttsHttp: OkHttpClient = baseHttp.newBuilder().readTimeout(240, TimeUnit.SECONDS).build()
        private val slowHttp = ttsHttp

        @Volatile private var instance: ServerClient? = null

        fun get(context: Context): ServerClient = instance ?: synchronized(this) {
            instance ?: ServerClient(context.applicationContext).also {
                instance = it
                it.watchNetworks()
            }
        }

        fun isConnectFailure(e: Throwable): Boolean = when (e) {
            is ServerHttpException -> false
            is ConnectException, is UnknownHostException, is NoRouteToHostException -> true
            is SocketTimeoutException -> e.message?.contains("connect", ignoreCase = true) == true
            else -> e.cause?.let { it !== e && isConnectFailure(it) } ?: false
        }

        fun ensureSuccess(resp: Response) {
            if (resp.isSuccessful) return
            val detail = try {
                JSONObject(resp.body?.string().orEmpty()).optString("error")
            } catch (_: Exception) {
                ""
            }
            throw ServerHttpException(resp.code, detail.ifBlank { "HTTP ${resp.code}" })
        }

        fun parseEntries(arr: JSONArray): List<PronEntry> = (0 until arr.length()).map { parseEntry(arr.getJSONObject(it)) }

        fun parseEntry(e: JSONObject) = PronEntry(
            id = e.optString("id"),
            term = e.optString("term"),
            respelling = e.optString("respelling"),
            phonemes = e.optString("phonemes"),
            caseSensitive = e.optBoolean("case_sensitive"),
            book = e.optStringOrNull("book"),
        )
    }
}

fun JSONObject.optStringOrNull(key: String): String? =
    if (isNull(key)) null else optString(key).takeIf { it.isNotBlank() && it != "null" }
