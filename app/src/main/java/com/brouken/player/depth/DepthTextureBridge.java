package com.brouken.player.depth;

import android.graphics.Bitmap;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Small helper that keeps the most recent depth texture and makes it available
 * to whichever stereo shader is driving the render pipeline.
 */
public class DepthTextureBridge {

    private final AtomicReference<Bitmap> latestDepthTexture = new AtomicReference<>();
    private DepthTextureListener listener;

    public void updateDepthFrame(Bitmap depthBitmap) {
        latestDepthTexture.set(depthBitmap);
        if (listener != null) {
            listener.onDepthTextureUpdated(depthBitmap);
        }
    }

    public Bitmap getLatestDepthTexture() {
        return latestDepthTexture.get();
    }

    public void setListener(DepthTextureListener listener) {
        this.listener = listener;
    }

    public interface DepthTextureListener {
        void onDepthTextureUpdated(Bitmap depthBitmap);
    }
}
