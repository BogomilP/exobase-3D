package com.brouken.player.depth;

/**
 * Exponential moving average smoother for depth maps.
 */
public class DepthTemporalSmoother {

    private float[] lastFrame;

    /**
     * Blend the current frame with the previous one to reduce flicker.
     */
    public float[] smooth(float[] current, int width, int height, float strength) {
        if (current == null) {
            return null;
        }

        if (lastFrame == null || lastFrame.length != current.length) {
            lastFrame = new float[current.length];
            System.arraycopy(current, 0, lastFrame, 0, current.length);
            return lastFrame;
        }

        final float keep = 1f - strength;
        for (int i = 0; i < current.length; i++) {
            lastFrame[i] = keep * lastFrame[i] + strength * current[i];
        }
        return lastFrame;
    }
}
