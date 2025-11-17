package com.brouken.player.depth;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.PixelCopy;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;

import androidx.annotation.Nullable;
import androidx.media3.ui.PlayerView;

/**
 * Grabs frames from the video surface, runs depth inference at reduced resolution
 * and feeds an upscaled depth texture to the stereo shader bridge.
 */
public class DepthPipeline {

    private static final String TAG = "DepthPipeline";
    private static final long CAPTURE_INTERVAL_MS = 50L;

    private final PlayerView playerView;
    private final DepthInferenceEngine engine;
    private final DepthTextureBridge bridge;
    private final HandlerThread captureThread = new HandlerThread("depth-capture");
    private final Handler captureHandler;

    private boolean running;

    public DepthPipeline(Context context, PlayerView playerView, DepthTextureBridge bridge) {
        this.playerView = playerView;
        this.bridge = bridge;
        this.engine = new DepthInferenceEngine(context);
        captureThread.start();
        captureHandler = new Handler(captureThread.getLooper());
    }

    public void setPlaying(boolean playing) {
        if (playing) {
            if (!running) {
                running = true;
                scheduleCapture();
            }
        } else {
            running = false;
            captureHandler.removeCallbacksAndMessages(null);
        }
    }

    public void release() {
        running = false;
        captureHandler.removeCallbacksAndMessages(null);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR2) {
            captureThread.quitSafely();
        } else {
            captureThread.quit();
        }
    }

    private void scheduleCapture() {
        captureHandler.postDelayed(this::captureFrame, CAPTURE_INTERVAL_MS);
    }

    private void captureFrame() {
        if (!running) {
            return;
        }

        final View videoSurface = playerView.getVideoSurfaceView();
        if (videoSurface == null) {
            scheduleCapture();
            return;
        }

        final int width = videoSurface.getWidth();
        final int height = videoSurface.getHeight();
        if (width <= 0 || height <= 0) {
            scheduleCapture();
            return;
        }

        final Bitmap captureBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N && videoSurface instanceof SurfaceView) {
            PixelCopy.request((SurfaceView) videoSurface, captureBitmap, result -> {
                if (result == PixelCopy.SUCCESS) {
                    process(captureBitmap, width, height);
                }
                scheduleCapture();
            }, captureHandler);
        } else if (videoSurface instanceof TextureView) {
            TextureView textureView = (TextureView) videoSurface;
            Bitmap bitmap = textureView.getBitmap(captureBitmap);
            process(bitmap, width, height);
            scheduleCapture();
        } else {
            Log.w(TAG, "Unsupported surface for depth capture");
            scheduleCapture();
        }
    }

    private void process(@Nullable Bitmap bitmap, int surfaceWidth, int surfaceHeight) {
        if (bitmap == null) {
            return;
        }

        final int maxDim = Math.max(bitmap.getWidth(), bitmap.getHeight());
        final float scale = maxDim > 320 ? 320f / maxDim : 1f;
        final int targetWidth = Math.max(1, Math.round(bitmap.getWidth() * scale));
        final int targetHeight = Math.max(1, Math.round(bitmap.getHeight() * scale));
        final Bitmap scaled = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true);

        Bitmap depthBitmap = engine.runInference(scaled, surfaceWidth, surfaceHeight);
        if (depthBitmap != null) {
            bridge.updateDepthFrame(depthBitmap);
        }
        if (scaled != bitmap) {
            scaled.recycle();
        }
        bitmap.recycle();
    }
}
