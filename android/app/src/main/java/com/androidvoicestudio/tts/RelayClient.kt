package com.androidvoicestudio.tts

import okhttp3.Call
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

data class RelayVoice(val id: String, val name: String, val language: String?)

data class SynthesisParams(
    val text: String,
    val voice: String?,
    val language: String?,
    val speed: Float,
    val instruct: String?,
)

/** HTTP error from the relay, carrying its status code and `{"error": ...}` detail. */
class RelayHttpException(val code: Int, message: String) : IOException(message)

/** Talks to the PC relay (server/vs_relay). All calls are blocking; use a background thread. */
class RelayClient(baseUrl: String, private val token: String) {
    private val base = normalizeServerUrl(baseUrl)

    private fun request(path: String) = Request.Builder()
        .url(base + path)
        .header("Authorization", "Bearer $token")

    /** A call for `POST /api/tts`; the response body is raw PCM16LE mono. */
    fun ttsCall(params: SynthesisParams): Call {
        val json = JSONObject()
            .put("text", params.text)
            .put("speed", params.speed.toDouble())
        params.voice?.takeIf { it.isNotBlank() }?.let { json.put("voice", it) }
        params.language?.takeIf { it.isNotBlank() }?.let { json.put("language", it) }
        params.instruct?.takeIf { it.isNotBlank() }?.let { json.put("instruct", it) }
        val body = json.toString().toRequestBody(JSON)
        return http.newCall(request("/api/tts").post(body).build())
    }

    fun health(): JSONObject = getJson("/api/health")

    fun voices(): List<RelayVoice> {
        val arr = getJson("/api/voices").getJSONArray("voices")
        return (0 until arr.length()).map { i ->
            val v = arr.getJSONObject(i)
            RelayVoice(
                id = v.optString("id"),
                name = v.optString("name", v.optString("id")),
                language = v.optString("language").takeIf { it.isNotBlank() && it != "null" },
            )
        }
    }

    private fun getJson(path: String): JSONObject {
        settingsHttp.newCall(request(path).get().build()).execute().use { resp ->
            ensureSuccess(resp)
            return JSONObject(resp.body?.string().orEmpty())
        }
    }

    companion object {
        private val JSON = "application/json; charset=utf-8".toMediaType()

        /**
         * Short connect timeout so an unreachable PC fails quickly; long read timeout
         * because the first sentence may wait for VoiceStudio to warm up (the relay
         * gives up first and reports why).
         */
        val http: OkHttpClient = OkHttpClient.Builder()
            .connectTimeout(4, TimeUnit.SECONDS)
            .writeTimeout(10, TimeUnit.SECONDS)
            .readTimeout(75, TimeUnit.SECONDS)
            .build()

        private val settingsHttp: OkHttpClient = http.newBuilder()
            .readTimeout(15, TimeUnit.SECONDS)
            .build()

        fun ensureSuccess(resp: Response) {
            if (resp.isSuccessful) return
            val detail = try {
                JSONObject(resp.body?.string().orEmpty()).optString("error")
            } catch (_: Exception) {
                ""
            }
            throw RelayHttpException(resp.code, detail.ifBlank { "HTTP ${resp.code}" })
        }
    }
}
