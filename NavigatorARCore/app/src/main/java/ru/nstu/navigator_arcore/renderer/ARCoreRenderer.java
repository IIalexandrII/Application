package ru.nstu.navigator_arcore.renderer;

import android.app.Activity;
import android.content.Context;
import android.media.Image;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.util.Log;
import android.widget.TextView;

import com.google.ar.core.Coordinates2d;
import com.google.ar.core.Frame;
import com.google.ar.core.Session;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import ru.nstu.navigator_arcore.Model;
import ru.nstu.navigator_arcore.R;
import ru.nstu.navigator_arcore.tools.BoundingBox;
import ru.nstu.navigator_arcore.tools.OverlayView;

public class ARCoreRenderer implements GLSurfaceView.Renderer {
    //test
    private long arLastTimeNs = 0;
    private int arFrameCount = 0;
    private float arFps = 0f;
    private TextView textFPS;
    private String fpsSTR = "ARCore FPS: %.1f | YOLO FPS: %.1f";

    // Inference FPS
    private long mdLastTimeNs = 0;
    private int mdFrameCount = 0;
    private volatile float mdFps = 0f;
    //-----------------------

    private final String TAG = "LOG.E";
    private boolean cameraTextureSet = false;

    private int viewportWidth = 1;
    private int viewportHeight = 1;

    private BackgroundRenderer backgroundRenderer;
    private Session session;
    private final Context context;
    private Model model;
    private OverlayView overlay;

    private volatile List<BoundingBox> lastBoxesImage = new ArrayList<>();
    private int depthFrameCounter = 0;
    private volatile boolean inferenceBusy = false;
    private final ExecutorService inferenceExecutor = Executors.newSingleThreadExecutor();

    public ARCoreRenderer(OverlayView overlay, Context context){
        this.context = context;
        this.overlay = overlay;
        this.textFPS = ((Activity) this.context).findViewById(R.id.FPSText);
    }

    public void setModel(Model model) {
        this.model = model;
    }

    public void setSession(Session session){
        this.session = session;
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        GLES20.glClearColor(0f, 0f, 0f, 1f);

        backgroundRenderer = new BackgroundRenderer();
        backgroundRenderer.createOnGlThread(context);

        cameraTextureSet = false;
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        viewportWidth = width;
        viewportHeight = height;
        GLES20.glViewport(0, 0, width, height);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        if (session == null) return;

        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);

        long nowNs = System.nanoTime();

        if (arLastTimeNs == 0) {
            arLastTimeNs = nowNs;
        }

        arFrameCount++;
        long diffNs = nowNs - arLastTimeNs;
        if (diffNs >= 1_000_000_000L) { // 1 сек
            arFps = arFrameCount * 1_000_000_000f / diffNs;
            arFrameCount = 0;
            arLastTimeNs = nowNs;
        }

        try {
            // добавить FPS
            int rotation = ((Activity) context).getWindowManager().getDefaultDisplay().getRotation();
            session.setDisplayGeometry(rotation, viewportWidth, viewportHeight);

            Frame frame = session.update();

            if (!cameraTextureSet && backgroundRenderer != null) {
                session.setCameraTextureName(backgroundRenderer.getTextureId());
                cameraTextureSet = true;
            }

            if (backgroundRenderer != null) {
                backgroundRenderer.draw(frame);
            }
            if (model == null) return;


            //-----------------------------------------------------------------------------------
            // ----------- 1) Быстро считаем distance НА ЭТОМ кадре по lastBoxesImage -----------
            Image depthImage = null;
            try {
                depthImage = frame.acquireDepthImage(); // deprecated warning ok
//                depthImage = frame.acquireDepthImage16Bits();
            } catch (Exception ignored) {}

            // локальная копия списка, чтобы не словить гонки
            List<BoundingBox> boxesForThisFrame = lastBoxesImage;
            if (boxesForThisFrame == null) boxesForThisFrame = new ArrayList<>();

            if (depthImage != null && !boxesForThisFrame.isEmpty()) {
                // можно считать через кадр, если хочешь совсем экономно:
                boolean doDepthNow = ((depthFrameCounter++ % 2) == 0);

                if (doDepthNow) {
                    final int camW = frame.getCamera().getImageIntrinsics().getImageDimensions()[0];
                    final int camH = frame.getCamera().getImageIntrinsics().getImageDimensions()[1];

                    // если intrinsics вдруг не дали (редко), можно fallback на depth size скейлом,
                    // но обычно intrinsics норм.
                    for (BoundingBox b : boxesForThisFrame) {
                        Float d = estimateDistanceMetersImagePixelsFast(depthImage, camW, camH, b);

                        // EMA сглаживание по предыдущему значению В ЭТОМ ЖЕ объекте списка
                        if (d != null) {
                            Float prev = b.distanceMeters;
                            if (prev != null) {
                                float alpha = 0.35f;
                                d = prev * (1f - alpha) + d * alpha;
                            }
                            b.distanceMeters = d;
                        } else {
                            b.distanceMeters = null;
                        }
                    }
                }
                depthImage.close();
            } else {
                if (depthImage != null) depthImage.close();
            }

            // ----------- 2) Рисуем overlay: маппим IMAGE_PIXELS -> VIEW каждый кадр -----------
            if (!boxesForThisFrame.isEmpty()) {
                List<BoundingBox> viewBoxes = new ArrayList<>(boxesForThisFrame.size());
                for (BoundingBox b : boxesForThisFrame) {
                    viewBoxes.add(mapImagePixelsToView(frame, b));
                }

                ((Activity) context).runOnUiThread(() -> {
                    overlay.setResults(viewBoxes, model.Classes);
                });
            } else {
                // если боксов нет — можно очистить overlay
                ((Activity) context).runOnUiThread(() -> {
                    overlay.setResults(new ArrayList<>(), model.Classes);
                });
            }

            // ----------- 3) YOLO инференс (как раньше), но результат кладём в lastBoxesImage -----------
            Image cameraImage;
            try {
                cameraImage = frame.acquireCameraImage();
            } catch (Exception e) {
                return;
            }

            if (!inferenceBusy) {
                inferenceBusy = true;

                inferenceExecutor.execute(() -> {
                    try {
                        List<BoundingBox> boxesImg = model.analyzeImage(cameraImage, rotation);
                        lastBoxesImage = (boxesImg != null) ? boxesImg : new ArrayList<>();

                    } catch (Exception e) {
                        Log.e(TAG, "ARCoreRenderer (onDrawFrame) errorInference error", e);
                    } finally {
                        long now = System.nanoTime();
                        if (mdLastTimeNs == 0) {
                            mdLastTimeNs = now;
                        }

                        mdFrameCount++;
                        long diff = now - mdLastTimeNs;

                        if (diff >= 1_000_000_000L) {
                            mdFps = mdFrameCount * 1_000_000_000f / diff;
                            mdFrameCount = 0;
                            mdLastTimeNs = now;

                            ((Activity) context).runOnUiThread(() -> {
                                textFPS.setText(String.format(fpsSTR, arFps, mdFps));
                            });
                        }


                        inferenceBusy = false;
                        cameraImage.close();
                    }
                });

            } else {
                cameraImage.close();
            }

        } catch (Exception e) {
            Log.e(TAG, "ARCoreRenderer (onDrawFrame) error", e);
        }
    }

    private Float estimateDistanceMetersImagePixelsFast(Image depthImage, int camW, int camH, BoundingBox box) {
        if (depthImage == null || box == null || box.rect == null) return null;

        int depthW = depthImage.getWidth();
        int depthH = depthImage.getHeight();

        // Берём нижнюю часть бокса (как у тебя) — там обычно depth стабильнее
        float left = box.rect.left + box.rect.width() * 0.30f;
        float right = box.rect.left + box.rect.width() * 0.70f;
        float top = box.rect.top + box.rect.height() * 0.65f;
        float bottom = box.rect.top + box.rect.height() * 0.90f;

        left = clamp(left, 0, camW - 1);
        right = clamp(right, 0, camW - 1);
        top = clamp(top, 0, camH - 1);
        bottom = clamp(bottom, 0, camH - 1);

        if (right <= left || bottom <= top) return null;

        int grid = 3; // 3x3 = 9 точек (очень быстро)
        int count = 0;

        float sum = 0f;
        float min = Float.MAX_VALUE;
        float max = -Float.MAX_VALUE;

        for (int iy = 0; iy < grid; iy++) {
            float ty = (grid == 1) ? 0.5f : (iy / (float) (grid - 1));
            float yImg = top + ty * (bottom - top);

            for (int ix = 0; ix < grid; ix++) {
                float tx = (grid == 1) ? 0.5f : (ix / (float) (grid - 1));
                float xImg = left + tx * (right - left);

                int xd = (int) (xImg / (float) camW * (depthW - 1));
                int yd = (int) (yImg / (float) camH * (depthH - 1));

                if (xd < 0 || xd >= depthW) continue;
                if (yd < 0 || yd >= depthH) continue;

                int mm = depthMmAt(depthImage, xd, yd);
                if (mm <= 0) continue;

                float m = mm / 1000f;
                if (m < 0.10f || m > 20f) continue;

                sum += m;
                min = Math.min(min, m);
                max = Math.max(max, m);
                count++;
            }
        }

        // мало точек — лучше ничего не показывать
        if (count < 5) return null;

        // trimmed mean: отбрасываем min/max (сильно режет выбросы)
        if (count >= 3) {
            sum = sum - min - max;
            count = count - 2;
        }

        float d = sum / (float) count;
        if (d < 0.10f || d > 20f) return null;

        return d;
    }

    // Читаю depth в миллиметрах из raw буфера (16-bit little-endian)
    private static int depthMmAt(Image depthImage, int x, int y) {
        Image.Plane plane = depthImage.getPlanes()[0];
        java.nio.ByteBuffer buffer = plane.getBuffer();
        int rowStride = plane.getRowStride();
        int pixelStride = plane.getPixelStride();

        int index = y * rowStride + x * pixelStride;
        int lo = buffer.get(index) & 0xFF;
        int hi = buffer.get(index + 1) & 0xFF;
        return (hi << 8) | lo;
    }

    private static float clamp(float v, float lo, float hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private BoundingBox mapImagePixelsToView(Frame frame, BoundingBox imgBox) {
        float[] in = new float[]{ imgBox.rect.left, imgBox.rect.top, imgBox.rect.right, imgBox.rect.bottom };
        float[] out = new float[4];

        frame.transformCoordinates2d(Coordinates2d.IMAGE_PIXELS, in, Coordinates2d.VIEW, out);

        float left = out[0], top = out[1], right = out[2], bottom = out[3];
        if (left > right) { float t = left; left = right; right = t; }
        if (top > bottom) { float t = top; top = bottom; bottom = t; }

        BoundingBox b = new BoundingBox(new android.graphics.RectF(left, top, right, bottom), imgBox.score, imgBox.clazz);
        b.distanceMeters = imgBox.distanceMeters;
        return b;
    }
}
