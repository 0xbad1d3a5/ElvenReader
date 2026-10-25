package com.androidreader

import android.app.Application
import com.androidreader.drama.Dramatization
import com.androidreader.player.PlaybackHub
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

class AndroidReaderApp : Application() {
    override fun onCreate() {
        super.onCreate()
        PlaybackHub.init(this)
        Dramatization.init(this)
    }

    companion object {
        /** Work that must outlive a screen, such as importing a book. */
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    }
}
