package com.brouken.player.depth;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Build;
import android.util.Log;

import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

/**
 * Runs a monocular depth model with TFLite/NNAPI and returns a normalized depth bitmap.
 */
public class DepthInferenceEngine {

    private static final String TAG = "DepthInferenceEngine";
    private static final String DEFAULT_MODEL = "depth_model.tflite";

    private final Context context;
    private final DepthTemporalSmoother smoother = new DepthTemporalSmoother();
    private Interpreter interpreter;
    private int inputWidth = 256;
    private int inputHeight = 256;

    public DepthInferenceEngine(Context context) {
        this.context = context.getApplicationContext();
        tryLoadInterpreter();
    }

    private void tryLoadInterpreter() {
        try {
            final MappedByteBuffer model = loadModel(DEFAULT_MODEL);
            if (model == null) {
                Log.w(TAG, "Depth model not bundled; falling back to luminance heuristic");
                return;
            }

            final Interpreter.Options options = new Interpreter.Options();
            Delegate nnapi = null;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                nnapi = new NnApiDelegate();
                options.addDelegate(nnapi);
            }
            interpreter = new Interpreter(model, options);

            final int[] inputShape = interpreter.getInputTensor(0).shape();
            if (inputShape.length >= 3) {
                inputHeight = inputShape[inputShape.length - 3];
                inputWidth = inputShape[inputShape.length - 2];
            }
        } catch (Throwable t) {
            Log.w(TAG, "Unable to initialize TFLite, using fallback", t);
            interpreter = null;
        }
    }

    private MappedByteBuffer loadModel(String assetName) {
        try (AssetFileDescriptor fileDescriptor = context.getAssets().openFd(assetName)) {
            try (FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
                final FileChannel fileChannel = inputStream.getChannel();
                final long startOffset = fileDescriptor.getStartOffset();
                final long declaredLength = fileDescriptor.getDeclaredLength();
                return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
            }
        } catch (IOException e) {
            return null;
        }
    }

    public Bitmap runInference(Bitmap frame, int outputWidth, int outputHeight) {
        if (frame == null) {
            return null;
        }
        Bitmap resized = Bitmap.createScaledBitmap(frame, inputWidth, inputHeight, true);
        float[] depthValues = interpreter != null
                ? runWithInterpreter(resized)
                : fallbackDepth(resized);

        float[] smoothed = smoother.smooth(depthValues, inputWidth, inputHeight, 0.2f);
        Bitmap depthBitmap = depthToBitmap(smoothed, inputWidth, inputHeight);
        return Bitmap.createScaledBitmap(depthBitmap, outputWidth, outputHeight, true);
    }

    private float[] runWithInterpreter(Bitmap frame) {
        final int channels = 3;
        final float[] input = new float[inputWidth * inputHeight * channels];
        int index = 0;
        for (int y = 0; y < inputHeight; y++) {
            for (int x = 0; x < inputWidth; x++) {
                final int pixel = frame.getPixel(x, y);
                input[index++] = ((pixel >> 16) & 0xff) / 255f;
                input[index++] = ((pixel >> 8) & 0xff) / 255f;
                input[index++] = (pixel & 0xff) / 255f;
            }
        }

        final int[] outputShape = interpreter.getOutputTensor(0).shape();
        final int outHeight = outputShape[outputShape.length - 3];
        final int outWidth = outputShape[outputShape.length - 2];
        final float[] output = new float[outWidth * outHeight];

        final float[][][][] inputBuffer = new float[1][inputHeight][inputWidth][channels];
        for (int i = 0; i < input.length; i++) {
            int y = (i / channels) / inputWidth;
            int x = (i / channels) % inputWidth;
            int c = i % channels;
            inputBuffer[0][y][x][c] = input[i];
        }

        final float[][][] outputBuffer = new float[1][outHeight][outWidth];
        interpreter.run(inputBuffer, outputBuffer);

        int outputIndex = 0;
        for (float[][] rows : outputBuffer) {
            for (float[] row : rows) {
                for (float value : row) {
                    output[outputIndex++] = value;
                }
            }
        }
        return normalize(output);
    }

    private float[] fallbackDepth(Bitmap frame) {
        final float[] luminance = new float[inputWidth * inputHeight];
        int index = 0;
        for (int y = 0; y < inputHeight; y++) {
            for (int x = 0; x < inputWidth; x++) {
                final int pixel = frame.getPixel(x, y);
                final float r = ((pixel >> 16) & 0xff) / 255f;
                final float g = ((pixel >> 8) & 0xff) / 255f;
                final float b = (pixel & 0xff) / 255f;
                luminance[index++] = (r + g + b) / 3f;
            }
        }
        return normalize(luminance);
    }

    private float[] normalize(float[] values) {
        final float min = Arrays.stream(values).min().orElse(0f);
        final float max = Arrays.stream(values).max().orElse(1f);
        final float range = Math.max(1e-3f, max - min);
        final float[] normalized = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            normalized[i] = (values[i] - min) / range;
        }
        return normalized;
    }

    private Bitmap depthToBitmap(float[] depth, int width, int height) {
        final int[] pixels = new int[depth.length];
        for (int i = 0; i < depth.length; i++) {
            int gray = (int) (depth[i] * 255f);
            gray = Math.max(0, Math.min(255, gray));
            pixels[i] = 0xff000000 | (gray << 16) | (gray << 8) | gray;
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888);
    }
}
