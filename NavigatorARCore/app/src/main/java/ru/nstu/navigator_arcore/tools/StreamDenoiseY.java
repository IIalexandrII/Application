package ru.nstu.navigator_arcore.tools;

import java.util.Arrays;

/**
 * StreamDenoiseY — потоковая обработка яркости (Y)
 *
 * Вход:  yIn  — плотный Y (w*h), 0..255
 * Выход: yOut — плотный Y (w*h), 0..255 (возвращается внутренний буфер, не менять снаружи)
 *
 */
public final class StreamDenoiseY {

    public enum Mode { NONE, DEHAZE, DERAIN, DESNOW }

    public static final class Config {
        public int N = 5;

        // highpass residual
        public float sigma = 1.2f;

        // DCT tile features
        public int tile = 16;
        public int lowK = 3;

        // thresholds / rules
        public float R_fog_thr = 1.2f;
        public float V_thr = 22.0f;
        public float A_thr = 0.12f;

        // dehaze (wavelet-like)
        public float gain = 1.25f;
        public float clipVal = 18.0f;
        public float cAContrast = 1.05f; // как у вас (cA - mean)*1.05 + mean

        // derain/desnow
        public float alpha = 0.85f;
        public float snowThr = 10.0f;

        public int tileStep = 1;
        public int modeEveryK = 1;

        public int modeHoldFrames = 8;
    }

    private final Config cfg;

    // size-dependent buffers
    private int w = 0, h = 0, wh = 0;

    // residual ring buffer: [N][w*h]
    private float[][] resBuf;
    private int resIdx = 0;
    private int filled = 0;

    // working buffers (reused)
    private float[] yF;        // float Y
    private float[] blur;      // Gaussian blur output
    private float[] r_t;       // residual current
    private float[] r_med;     // per-pixel median residual
    private byte[] yOut;       // output Y

    // DWT buffers (reused) - 1-level Haar
    private float[] dwtTmp;    // intermediate
    private float[] cA, cH, cV, cD; // subbands (sizes ~ (w/2)*(h/2))
    private float[] dwtRecon;  // reconstructed

    // DCT precompute for tile size
    private float[][] dctC;    // [tile][tile]
    private float[][] dctCT;   // transpose
    private float[] dctBlock;  // tile*tile
    private float[] dctTemp;   // tile*tile
    private float[] dctOut;    // tile*tile

    // mode smoothing state
    private Mode modePrev = Mode.NONE;
    private int modeHold = 0;
    private long frameCount = 0;

    // last metrics
    private volatile Mode lastMode = Mode.NONE;
    private volatile float lastR = 0f, lastA = 0f, lastV = 0f;
    private volatile float lastMs = 0f;

    public StreamDenoiseY(Config cfg) {
        this.cfg = (cfg != null) ? cfg : new Config();
        if (this.cfg.N < 1) this.cfg.N = 1;
        if (this.cfg.tile < 2) this.cfg.tile = 16;
        if (this.cfg.lowK < 1) this.cfg.lowK = 3;
        if (this.cfg.tileStep < 1) this.cfg.tileStep = 1;
        if (this.cfg.modeEveryK < 1) this.cfg.modeEveryK = 1;
        if (this.cfg.modeHoldFrames < 0) this.cfg.modeHoldFrames = 0;
    }

    // --- public API ---

    /** Обработать один кадр яркости Y (плотный w*h). Возвращает внутренний буфер yOut. */
    public byte[] process(byte[] yIn, int width, int height) {
        long t0 = System.nanoTime();

        ensureSize(width, height);

        // yIn -> yF
        for (int i = 0; i < wh; i++) {
            yF[i] = (float) (yIn[i] & 0xFF);
        }

        // residual r_t = y - blur(y)
        gaussianBlur(yF, blur, w, h, cfg.sigma);
        for (int i = 0; i < wh; i++) {
            r_t[i] = yF[i] - blur[i];
        }

        // push residual into ring buffer
        pushResidual(r_t);

        // пока буфер не заполнен — возвращаем исходный Y
        if (filled < cfg.N) {
            System.arraycopy(yIn, 0, yOut, 0, wh);
            lastMode = Mode.NONE;
            lastR = lastA = lastV = 0f;
            lastMs = (System.nanoTime() - t0) / 1_000_000f;
            return yOut;
        }

        // mode selection: можно считать не каждый кадр
        Mode mode = lastMode;
        boolean recalcMode = (frameCount % cfg.modeEveryK == 0);
        float R_ratio = lastR, A_aniso = lastA, V_temp = lastV;

        if (recalcMode) {
            DctFeatures feat = dctTileFeatures(yIn, w, h, cfg.tile, cfg.lowK, cfg.tileStep);
            R_ratio = feat.rRatio;
            A_aniso = feat.aAniso;
            V_temp = temporalVariance(resBuf, cfg.N, wh);

            mode = pickMode(R_ratio, A_aniso, V_temp, cfg.R_fog_thr, cfg.V_thr, cfg.A_thr);
            // удержание режима
            if (mode != modePrev) {
                if (modeHold < cfg.modeHoldFrames) {
                    mode = modePrev;
                    modeHold++;
                } else {
                    modePrev = mode;
                    modeHold = 0;
                }
            } else {
                modeHold = 0;
            }
        }

        // temporal median residual (per pixel)
        temporalMedianResidual(resBuf, cfg.N, wh, r_med);

        // apply mode
        switch (mode) {
            case DEHAZE: {
                // wavelet_dehaze_like on Y
                waveletDehazeLike(yIn, w, h, cfg.gain, cfg.clipVal, cfg.cAContrast, yOut);
                break;
            }
            case DERAIN: {
                derainDesnowTemporalMedian(yIn, r_t, r_med, w, h, cfg.alpha, false, cfg.snowThr, yOut);
                break;
            }
            case DESNOW: {
                derainDesnowTemporalMedian(yIn, r_t, r_med, w, h, cfg.alpha, true, cfg.snowThr, yOut);
                break;
            }
            default: {
                System.arraycopy(yIn, 0, yOut, 0, wh);
                break;
            }
        }

        lastMode = mode;
        lastR = R_ratio;
        lastA = A_aniso;
        lastV = V_temp;
        lastMs = (System.nanoTime() - t0) / 1_000_000f;

        frameCount++;
        return yOut;
    }

    public Mode getLastMode() { return lastMode; }
    public float getLastRratio() { return lastR; }
    public float getLastAaniso() { return lastA; }
    public float getLastVtemp() { return lastV; }
    public float getLastMs() { return lastMs; }

    // --- core internals ---

    private void ensureSize(int width, int height) {
        if (width <= 0 || height <= 0) throw new IllegalArgumentException("Invalid size");
        if (width == w && height == h) return;

        w = width; h = height; wh = w * h;

        // ring buffer
        resBuf = new float[cfg.N][wh];
        resIdx = 0;
        filled = 0;

        // working buffers
        yF = new float[wh];
        blur = new float[wh];
        r_t = new float[wh];
        r_med = new float[wh];
        yOut = new byte[wh];

        // DWT buffers: use even dims for subbands
        int w2 = w / 2;
        int h2 = h / 2;
        int wh2 = Math.max(1, w2 * h2);

        dwtTmp = new float[wh];     // for row/col transforms
        dwtRecon = new float[wh];   // reconstruction

        cA = new float[wh2];
        cH = new float[wh2];
        cV = new float[wh2];
        cD = new float[wh2];

        // DCT precompute for current tile size
        int t = cfg.tile;
        dctC = new float[t][t];
        dctCT = new float[t][t];
        buildDctMatrix(t, dctC, dctCT);

        dctBlock = new float[t * t];
        dctTemp = new float[t * t];
        dctOut = new float[t * t];
    }

    private void pushResidual(float[] r) {
        // copy current residual into ring slot
        System.arraycopy(r, 0, resBuf[resIdx], 0, wh);
        resIdx = (resIdx + 1) % cfg.N;
        if (filled < cfg.N) filled++;
    }

    // ----- Gaussian blur (separable) -----

    private static void gaussianBlur(float[] src, float[] dst, int w, int h, float sigma) {
        if (sigma <= 0f) {
            System.arraycopy(src, 0, dst, 0, w * h);
            return;
        }
        int radius = (int) Math.ceil(3.0 * sigma);
        int kSize = radius * 2 + 1;

        float[] kernel = new float[kSize];
        float sum = 0f;
        float inv2s2 = 1f / (2f * sigma * sigma);
        for (int i = -radius; i <= radius; i++) {
            float v = (float) Math.exp(-(i * i) * inv2s2);
            kernel[i + radius] = v;
            sum += v;
        }
        for (int i = 0; i < kSize; i++) kernel[i] /= sum;

        float[] tmp = new float[w * h];

        // horizontal
        for (int y = 0; y < h; y++) {
            int row = y * w;
            for (int x = 0; x < w; x++) {
                float acc = 0f;
                for (int k = -radius; k <= radius; k++) {
                    int xx = x + k;
                    if (xx < 0) xx = 0;
                    else if (xx >= w) xx = w - 1;
                    acc += src[row + xx] * kernel[k + radius];
                }
                tmp[row + x] = acc;
            }
        }

        // vertical
        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                float acc = 0f;
                for (int k = -radius; k <= radius; k++) {
                    int yy = y + k;
                    if (yy < 0) yy = 0;
                    else if (yy >= h) yy = h - 1;
                    acc += tmp[yy * w + x] * kernel[k + radius];
                }
                dst[y * w + x] = acc;
            }
        }
    }

    // ----- DCT tile features -----

    private static final class DctFeatures {
        final float rRatio;
        final float aAniso;
        DctFeatures(float r, float a) { this.rRatio = r; this.aAniso = a; }
    }

    private DctFeatures dctTileFeatures(byte[] y, int w, int h, int tile, int lowK, int tileStep) {
        int hh = (h / tile) * tile;
        int ww = (w / tile) * tile;
        if (hh <= 0 || ww <= 0) return new DctFeatures(0f, 0f);

        final float eps = 1e-6f;
        double sumRatio = 0.0;
        double sumAnis = 0.0;
        int count = 0;

        for (int i = 0; i < hh; i += tile * tileStep) {
            for (int j = 0; j < ww; j += tile * tileStep) {

                // load block and subtract mean
                float mean = 0f;
                int idx = 0;
                for (int yy = 0; yy < tile; yy++) {
                    int base = (i + yy) * w + j;
                    for (int xx = 0; xx < tile; xx++) {
                        float v = (float) (y[base + xx] & 0xFF);
                        dctBlock[idx++] = v;
                        mean += v;
                    }
                }
                mean /= (tile * tile);
                for (int k = 0; k < dctBlock.length; k++) dctBlock[k] -= mean;

                // 2D DCT: out = C * block * C^T
                dct2(dctBlock, dctTemp, dctOut, dctC, dctCT, tile);

                // energies
                double eLow = 0.0;
                for (int a = 0; a < lowK; a++) {
                    for (int b = 0; b < lowK; b++) {
                        float v = dctOut[a * tile + b];
                        eLow += (double) v * v;
                    }
                }

                double eHigh = 0.0;
                double eVert = 0.0; // d[:, lowK:]
                double eHorz = 0.0; // d[lowK:, :]

                for (int a = 0; a < tile; a++) {
                    for (int b = 0; b < tile; b++) {
                        float v = dctOut[a * tile + b];
                        double vv = (double) v * v;

                        if (!(a < lowK && b < lowK)) eHigh += vv;
                        if (b >= lowK) eVert += vv;
                        if (a >= lowK) eHorz += vv;
                    }
                }

                double ratio = eHigh / (eLow + eps);
                double anis = Math.abs(eHorz - eVert) / (eHorz + eVert + eps);

                sumRatio += ratio;
                sumAnis += anis;
                count++;
            }
        }

        if (count == 0) return new DctFeatures(0f, 0f);
        return new DctFeatures((float) (sumRatio / count), (float) (sumAnis / count));
    }

    private static void buildDctMatrix(int n, float[][] C, float[][] CT) {
        // DCT-II orthonormal
        final double invN = 1.0 / n;
        for (int u = 0; u < n; u++) {
            double alpha = (u == 0) ? Math.sqrt(invN) : Math.sqrt(2.0 * invN);
            for (int x = 0; x < n; x++) {
                double v = alpha * Math.cos((Math.PI * (2 * x + 1) * u) / (2.0 * n));
                C[u][x] = (float) v;
                CT[x][u] = (float) v;
            }
        }
    }

    private static void dct2(float[] block, float[] temp, float[] out, float[][] C, float[][] CT, int n) {
        // temp = C * block
        // block is row-major n*n
        for (int u = 0; u < n; u++) {
            for (int y = 0; y < n; y++) {
                float acc = 0f;
                int row = y * n;
                for (int x = 0; x < n; x++) {
                    acc += C[u][x] * block[row + x];
                }
                temp[u * n + y] = acc;
            }
        }
        // out = temp * C^T
        for (int u = 0; u < n; u++) {
            int tempRow = u * n;
            for (int v = 0; v < n; v++) {
                float acc = 0f;
                for (int y = 0; y < n; y++) {
                    acc += temp[tempRow + y] * CT[y][v];
                }
                out[u * n + v] = acc;
            }
        }
    }

    // ----- Temporal metrics -----

    private static float temporalVariance(float[][] resBuf, int N, int wh) {
        // mean var over pixels
        double sumVar = 0.0;

        for (int p = 0; p < wh; p++) {
            // mean
            double m = 0.0;
            for (int t = 0; t < N; t++) m += resBuf[t][p];
            m /= N;

            double v = 0.0;
            for (int t = 0; t < N; t++) {
                double d = resBuf[t][p] - m;
                v += d * d;
            }
            v /= N;
            sumVar += v;
        }
        return (float) (sumVar / wh);
    }

    private static void temporalMedianResidual(float[][] resBuf, int N, int wh, float[] out) {
        // оптимизировано под N=5 (как у вас). Для других N — fallback с сортировкой.
        if (N == 5) {
            for (int p = 0; p < wh; p++) {
                float a = resBuf[0][p];
                float b = resBuf[1][p];
                float c = resBuf[2][p];
                float d = resBuf[3][p];
                float e = resBuf[4][p];
                out[p] = median5(a, b, c, d, e);
            }
            return;
        }

        float[] tmp = new float[N];
        for (int p = 0; p < wh; p++) {
            for (int t = 0; t < N; t++) tmp[t] = resBuf[t][p];
            Arrays.sort(tmp);
            out[p] = tmp[N / 2];
        }
    }

    // median of 5 via sorting network-ish comparisons (fast, no arrays)
    private static float median5(float a, float b, float c, float d, float e) {
        // helper: ensure a<=b
        if (a > b) { float t=a; a=b; b=t; }
        if (c > d) { float t=c; c=d; d=t; }
        if (a > c) { float t=a; a=c; c=t; t=b; b=d; d=t; }
        if (b > e) { float t=b; b=e; e=t; }
        if (b > c) { float t=b; b=c; c=t; }
        if (d > e) { float t=d; d=e; e=t; }
        if (c > d) { float t=c; c=d; d=t; }
        // now c is median
        return c;
    }

    // ----- Mode rules -----

    private static Mode pickMode(float R_ratio, float A_aniso, float V_temp,
                                 float R_fog_thr, float V_thr, float A_thr) {
        if (R_ratio < R_fog_thr) return Mode.DEHAZE;
        if (V_temp > V_thr) return (A_aniso > A_thr) ? Mode.DERAIN : Mode.DESNOW;
        return Mode.NONE;
    }

    // ----- Filters -----

    private void derainDesnowTemporalMedian(byte[] y, float[] r_t, float[] r_med,
                                            int w, int h, float alpha,
                                            boolean snow, float thr,
                                            byte[] out) {
        int wh = w * h;
        for (int i = 0; i < wh; i++) {
            float delta = (r_t[i] - r_med[i]);
            if (snow) {
                if (Math.abs(r_t[i]) <= thr) delta = 0f;
            }
            float yHat = (float) (y[i] & 0xFF) - alpha * delta;
            int v = clamp255(Math.round(yHat));
            out[i] = (byte) v;
        }
    }

    /**
     * 1-level Haar DWT boost деталей с клипом (как wavelet_dehaze_like в Python).
     * Работает на Y, возвращает Y'.
     */
    private void waveletDehazeLike(byte[] y, int w, int h,
                                   float gain, float clipVal, float cAContrast,
                                   byte[] out) {

        // Для Haar 1 уровня нужны четные размеры. Если нечетные — обрабатываем четную область,
        // а последнюю строку/колонку копируем как есть.
        int wEven = (w / 2) * 2;
        int hEven = (h / 2) * 2;
        int w2 = wEven / 2;
        int h2 = hEven / 2;
        if (w2 <= 0 || h2 <= 0) {
            System.arraycopy(y, 0, out, 0, w * h);
            return;
        }

        // y -> dwtRecon (float) for processing region
        for (int yy = 0; yy < hEven; yy++) {
            int base = yy * w;
            for (int xx = 0; xx < wEven; xx++) {
                dwtRecon[base + xx] = (float) (y[base + xx] & 0xFF);
            }
        }

        // forward Haar DWT (1 level) into cA,cH,cV,cD
        haarDwt1Level(dwtRecon, w, h, wEven, hEven, cA, cH, cV, cD);

        // contrast on cA: (cA - mean)*cAContrast + mean
        float mean = 0f;
        int wh2 = w2 * h2;
        for (int i = 0; i < wh2; i++) mean += cA[i];
        mean /= wh2;
        for (int i = 0; i < wh2; i++) {
            cA[i] = (cA[i] - mean) * cAContrast + mean;
        }

        // boost details with clip
        for (int i = 0; i < wh2; i++) {
            cH[i] = clip(gain * cH[i], clipVal);
            cV[i] = clip(gain * cV[i], clipVal);
            cD[i] = clip(gain * cD[i], clipVal);
        }

        // inverse Haar DWT (1 level) back into dwtRecon
        haarIdwt1Level(dwtRecon, w, h, wEven, hEven, cA, cH, cV, cD);

        // write output: processed even region + copy borders if odd
        // processed region:
        for (int yy = 0; yy < hEven; yy++) {
            int base = yy * w;
            for (int xx = 0; xx < wEven; xx++) {
                int v = clamp255(Math.round(dwtRecon[base + xx]));
                out[base + xx] = (byte) v;
            }
        }
        // copy last col if odd width
        if (wEven != w) {
            for (int yy = 0; yy < hEven; yy++) {
                out[yy * w + (w - 1)] = y[yy * w + (w - 1)];
            }
        }
        // copy last row if odd height
        if (hEven != h) {
            int lastRow = (h - 1) * w;
            System.arraycopy(y, lastRow, out, lastRow, w);
        }
    }

    private static float clip(float v, float clipVal) {
        if (v > clipVal) return clipVal;
        if (v < -clipVal) return -clipVal;
        return v;
    }

    private static int clamp255(int v) {
        if (v < 0) return 0;
        if (v > 255) return 255;
        return v;
    }

    /**
     * Haar DWT 1-level:
     * cA,cH,cV,cD имеют размер (wEven/2)*(hEven/2).
     * Внутри используем стандартную разложение по 2x2 блокам:
     * a=(x00+x01+x10+x11)/2, h=(x00-x01+x10-x11)/2, v=(x00+x01-x10-x11)/2, d=(x00-x01-x10+x11)/2
     * (масштабирование подобрано "мягкое", близкое по ощущению к вашему питону).
     */
    private static void haarDwt1Level(float[] y, int w, int h, int wEven, int hEven,
                                      float[] cA, float[] cH, float[] cV, float[] cD) {
        int w2 = wEven / 2;
        int h2 = hEven / 2;
        int outIdx = 0;
        for (int yy = 0; yy < hEven; yy += 2) {
            int row0 = yy * w;
            int row1 = (yy + 1) * w;
            for (int xx = 0; xx < wEven; xx += 2) {
                float x00 = y[row0 + xx];
                float x01 = y[row0 + xx + 1];
                float x10 = y[row1 + xx];
                float x11 = y[row1 + xx + 1];

                float a = (x00 + x01 + x10 + x11) * 0.5f;
                float hF = (x00 - x01 + x10 - x11) * 0.5f;
                float vF = (x00 + x01 - x10 - x11) * 0.5f;
                float dF = (x00 - x01 - x10 + x11) * 0.5f;

                cA[outIdx] = a;
                cH[outIdx] = hF;
                cV[outIdx] = vF;
                cD[outIdx] = dF;
                outIdx++;
            }
        }
    }

    private static void haarIdwt1Level(float[] out, int w, int h, int wEven, int hEven,
                                       float[] cA, float[] cH, float[] cV, float[] cD) {
        int w2 = wEven / 2;
        int h2 = hEven / 2;

        int idx = 0;
        for (int yy = 0; yy < hEven; yy += 2) {
            int row0 = yy * w;
            int row1 = (yy + 1) * w;
            for (int xx = 0; xx < wEven; xx += 2) {
                float a = cA[idx];
                float hF = cH[idx];
                float vF = cV[idx];
                float dF = cD[idx];
                idx++;

                // inverse of the above (consistent scaling)
                float x00 = (a + hF + vF + dF) * 0.5f;
                float x01 = (a - hF + vF - dF) * 0.5f;
                float x10 = (a + hF - vF - dF) * 0.5f;
                float x11 = (a - hF - vF + dF) * 0.5f;

                out[row0 + xx] = x00;
                out[row0 + xx + 1] = x01;
                out[row1 + xx] = x10;
                out[row1 + xx + 1] = x11;
            }
        }
    }
}
