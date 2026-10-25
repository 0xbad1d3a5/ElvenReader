package com.androidreader

import android.app.Application
import com.androidreader.drama.Dramatization
import com.androidreader.player.PlaybackHub

class AndroidReaderApp : Application() {
    override fun onCreate() {
        super.onCreate()
        PlaybackHub.init(this)
        Dramatization.init(this)
    }
}
