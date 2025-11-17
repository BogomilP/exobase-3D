package com.brouken.player;

import android.content.Context;
import android.graphics.Color;
import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.GLUtils;
import android.util.AttributeSet;
import android.view.Surface;
import android.view.View;
import android.widget.FrameLayout;

import androidx.annotation.Nullable;
import androidx.media3.common.Player;
import androidx.media3.ui.AspectRatioFrameLayout;

import com.brouken.player.dtpv.DoubleTapPlayerView;
import com.brouken.player.depth.DepthTextureBridge;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * A container view that keeps the existing controller/gesture overlay while routing video frames
 * through a {@link GLSurfaceView}. The renderer feeds ExoPlayer via {@link SurfaceTexture} and
 * draws stereo quads with a small depth-based horizontal offset. Subtitles and black bars remain
 * on the default overlay layers provided by {@link DoubleTapPlayerView}.
 */
public class StereoGlPlayerView extends DoubleTapPlayerView {

    private final GLSurfaceView glSurfaceView;
    private final StereoRenderer stereoRenderer;
    @Nullable
    private SurfaceTexture surfaceTexture;
    @Nullable
    private DepthTextureBridge depthTextureBridge;

    public StereoGlPlayerView(Context context) {
        this(context, null);
    }

    public StereoGlPlayerView(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public StereoGlPlayerView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);

        setUseArtwork(false);
        setShutterBackgroundColor(Color.TRANSPARENT);

        stereoRenderer = new StereoRenderer();

        glSurfaceView = new GLSurfaceView(context);
        glSurfaceView.setClickable(false);
        glSurfaceView.setFocusable(false);
        glSurfaceView.setEGLContextClientVersion(2);
        glSurfaceView.setRenderer(stereoRenderer);
        glSurfaceView.setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);

        final AspectRatioFrameLayout contentFrame = findViewById(R.id.exo_content_frame);
        if (contentFrame != null) {
            contentFrame.addView(glSurfaceView, 0, new LayoutParams(
                    LayoutParams.MATCH_PARENT,
                    LayoutParams.MATCH_PARENT));
        }

        final FrameLayout overlayFrame = findViewById(R.id.exo_overlay);
        if (overlayFrame != null) {
            final FrameLayout blackBarLayer = new FrameLayout(context);
            blackBarLayer.setLayoutParams(new LayoutParams(
                    LayoutParams.MATCH_PARENT,
                    LayoutParams.MATCH_PARENT));
            blackBarLayer.setBackgroundColor(Color.TRANSPARENT);
            blackBarLayer.setClickable(false);
            overlayFrame.addView(blackBarLayer, 0);
        }
    }

    @Override
    public void setPlayer(@Nullable Player player) {
        super.setPlayer(player);
        stereoRenderer.setPlayer(player);
    }

    @Override
    public void setDepthTextureBridge(DepthTextureBridge bridge) {
        super.setDepthTextureBridge(bridge);
        this.depthTextureBridge = bridge;
        stereoRenderer.setDepthTextureBridge(bridge);
    }

    @Nullable
    public SurfaceTexture getVideoSurfaceTexture() {
        return surfaceTexture;
    }

    @Override
    public boolean setDepthFrameListener(@Nullable Runnable listener) {
        stereoRenderer.setDepthFrameListener(listener);
        return true;
    }

    @Nullable
    @Override
    public View getDepthCaptureSurface() {
        return glSurfaceView;
    }

    @Override
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        glSurfaceView.onResume();
    }

    @Override
    protected void onDetachedFromWindow() {
        stereoRenderer.release();
        glSurfaceView.onPause();
        super.onDetachedFromWindow();
    }

    private final class StereoRenderer implements GLSurfaceView.Renderer, SurfaceTexture.OnFrameAvailableListener {

        private static final String VERTEX_SHADER =
                "attribute vec4 aPosition;" +
                        "attribute vec2 aTexCoords;" +
                        "varying vec2 vTexCoords;" +
                        "void main() {" +
                        "  gl_Position = aPosition;" +
                        "  vTexCoords = aTexCoords;" +
                        "}"; 

        private static final String FRAGMENT_SHADER =
                "#extension GL_OES_EGL_image_external : require\n" +
                        "precision mediump float;" +
                        "varying vec2 vTexCoords;" +
                        "uniform samplerExternalOES uTexture;" +
                        "uniform sampler2D uDepthTexture;" +
                        "uniform float uEyeOffset;" +
                        "uniform float uMaxParallax;" +
                        "uniform vec2 uTexelSize;" +
                        "vec4 sampleColor(vec2 coords, vec2 direction) {" +
                        "  vec2 safe = clamp(coords, vec2(0.0), vec2(1.0));" +
                        "  vec4 base = texture2D(uTexture, safe);" +
                        "  vec4 ahead = texture2D(uTexture, clamp(safe + direction, vec2(0.0), vec2(1.0)));" +
                        "  vec4 behind = texture2D(uTexture, clamp(safe - direction, vec2(0.0), vec2(1.0)));" +
                        "  return base * 0.6 + ahead * 0.25 + behind * 0.15;" +
                        "}" +
                        "void main() {" +
                        "  float depth = texture2D(uDepthTexture, vTexCoords).r;" +
                        "  float parallax = clamp(uEyeOffset * (1.0 - depth), -uMaxParallax, uMaxParallax);" +
                        "  vec2 shifted = vTexCoords + vec2(parallax, 0.0);" +
                        "  vec2 direction = vec2(sign(parallax) * uTexelSize.x, 0.0);" +
                        "  gl_FragColor = sampleColor(shifted, direction);" +
                        "}"; 

        private static final float[] VERTICES = new float[]{
                -1f, -1f, 0f, 1f,
                1f, -1f, 1f, 1f,
                -1f, 1f, 0f, 0f,
                1f, 1f, 1f, 0f
        };

        private final FloatBuffer vertexBuffer;
        private SurfaceTexture surfaceTexture;
        private Surface surface;
        private int textureId = -1;
        private int program = -1;
        private int positionHandle;
        private int texCoordHandle;
        private int eyeOffsetHandle;
        private int textureHandle;
        private int depthTextureHandle;
        private int maxParallaxHandle;
        private int texelSizeHandle;
        private float depthOffset = 0.04f;
        private float maxParallax = 0.07f;
        private int depthTextureId = -1;
        private int depthWidth = 1;
        private int depthHeight = 1;
        @Nullable
        private DepthTextureBridge depthTextureBridge;
        private Player player;
        @Nullable
        private Runnable depthFrameListener;

        StereoRenderer() {
            vertexBuffer = ByteBuffer.allocateDirect(VERTICES.length * 4)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer();
            vertexBuffer.put(VERTICES).position(0);
        }

        void setPlayer(@Nullable Player player) {
            if (this.player == player) {
                return;
            }
            if (this.player != null && surface != null) {
                this.player.clearVideoSurface(surface);
            }
            this.player = player;
            if (player != null && surface != null) {
                player.setVideoSurface(surface);
            }
        }

        void setDepthFrameListener(@Nullable Runnable depthFrameListener) {
            this.depthFrameListener = depthFrameListener;
        }

        void setDepthTextureBridge(@Nullable DepthTextureBridge bridge) {
            if (depthTextureBridge == bridge) {
                return;
            }
            if (depthTextureBridge != null) {
                depthTextureBridge.setListener(null);
            }
            depthTextureBridge = bridge;
            if (bridge != null) {
                bridge.setListener(this::onDepthTextureUpdated);
                final Bitmap latest = bridge.getLatestDepthTexture();
                if (latest != null) {
                    onDepthTextureUpdated(latest);
                }
            }
        }

        void release() {
            if (player != null && surface != null) {
                player.clearVideoSurface(surface);
            }
            if (surfaceTexture != null) {
                surfaceTexture.release();
                surfaceTexture = null;
                StereoGlPlayerView.this.surfaceTexture = null;
            }
            if (surface != null) {
                surface.release();
                surface = null;
            }
            if (depthTextureBridge != null) {
                depthTextureBridge.setListener(null);
            }
            textureId = -1;
            program = -1;
        }

        private void onDepthTextureUpdated(Bitmap depthBitmap) {
            if (depthBitmap == null) {
                return;
            }
            glSurfaceView.queueEvent(() -> {
                uploadDepthTexture(depthBitmap);
                glSurfaceView.requestRender();
            });
        }

        @Override
        public void onSurfaceCreated(GL10 gl, EGLConfig config) {
            textureId = createExternalTexture();
            surfaceTexture = new SurfaceTexture(textureId);
            StereoGlPlayerView.this.surfaceTexture = surfaceTexture;
            surfaceTexture.setOnFrameAvailableListener(this);
            surface = new Surface(surfaceTexture);
            if (player != null) {
                player.setVideoSurface(surface);
            }
            program = buildProgram();
            depthTextureId = createDepthTexture();
            GLES20.glClearColor(0f, 0f, 0f, 1f);
        }

        @Override
        public void onSurfaceChanged(GL10 gl, int width, int height) {
            GLES20.glViewport(0, 0, width, height);
        }

        @Override
        public void onDrawFrame(GL10 gl) {
            if (surfaceTexture == null) {
                return;
            }
            surfaceTexture.updateTexImage();

            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

            GLES20.glUseProgram(program);
            vertexBuffer.position(0);
            GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 16, vertexBuffer);
            GLES20.glEnableVertexAttribArray(positionHandle);

            vertexBuffer.position(2);
            GLES20.glVertexAttribPointer(texCoordHandle, 2, GLES20.GL_FLOAT, false, 16, vertexBuffer);
            GLES20.glEnableVertexAttribArray(texCoordHandle);

            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);
            GLES20.glUniform1i(textureHandle, 0);

            GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, depthTextureId);
            GLES20.glUniform1i(depthTextureHandle, 1);
            GLES20.glUniform1f(maxParallaxHandle, maxParallax);
            GLES20.glUniform2f(texelSizeHandle, 1f / Math.max(1, depthWidth), 1f / Math.max(1, depthHeight));

            // Left eye
            GLES20.glUniform1f(eyeOffsetHandle, -depthOffset);
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

            // Right eye
            GLES20.glUniform1f(eyeOffsetHandle, depthOffset);
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        }

        @Override
        public void onFrameAvailable(SurfaceTexture surfaceTexture) {
            glSurfaceView.requestRender();
            if (depthFrameListener != null) {
                depthFrameListener.run();
            }
        }

        private void uploadDepthTexture(Bitmap depthBitmap) {
            if (depthTextureId == -1) {
                depthTextureId = createDepthTexture();
            }
            if (depthTextureId == -1) {
                return;
            }
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, depthTextureId);
            depthWidth = depthBitmap.getWidth();
            depthHeight = depthBitmap.getHeight();
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, depthBitmap, 0);
        }

        private int buildProgram() {
            final int vertexShader = compileShader(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER);
            final int fragmentShader = compileShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
            final int programId = GLES20.glCreateProgram();
            GLES20.glAttachShader(programId, vertexShader);
            GLES20.glAttachShader(programId, fragmentShader);
            GLES20.glLinkProgram(programId);
            positionHandle = GLES20.glGetAttribLocation(programId, "aPosition");
            texCoordHandle = GLES20.glGetAttribLocation(programId, "aTexCoords");
            eyeOffsetHandle = GLES20.glGetUniformLocation(programId, "uEyeOffset");
            textureHandle = GLES20.glGetUniformLocation(programId, "uTexture");
            depthTextureHandle = GLES20.glGetUniformLocation(programId, "uDepthTexture");
            maxParallaxHandle = GLES20.glGetUniformLocation(programId, "uMaxParallax");
            texelSizeHandle = GLES20.glGetUniformLocation(programId, "uTexelSize");
            return programId;
        }

        private int compileShader(int type, String code) {
            final int shader = GLES20.glCreateShader(type);
            GLES20.glShaderSource(shader, code);
            GLES20.glCompileShader(shader);
            return shader;
        }

        private int createExternalTexture() {
            final int[] textures = new int[1];
            GLES20.glGenTextures(1, textures, 0);
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textures[0]);
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
            return textures[0];
        }

        private int createDepthTexture() {
            final int[] textures = new int[1];
            GLES20.glGenTextures(1, textures, 0);
            if (textures[0] == 0) {
                return -1;
            }
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textures[0]);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

            final ByteBuffer buffer = ByteBuffer.allocateDirect(4);
            buffer.put((byte) 128).put((byte) 128).put((byte) 128).put((byte) 255).position(0);
            GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, 1, 1, 0,
                    GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, buffer);
            depthWidth = 1;
            depthHeight = 1;
            return textures[0];
        }
    }
}
