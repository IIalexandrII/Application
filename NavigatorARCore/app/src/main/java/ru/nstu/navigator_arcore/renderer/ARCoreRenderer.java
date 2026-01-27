package ru.nstu.navigator_arcore.renderer;

import android.content.Context;
import android.media.Image;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.util.Log;
import android.view.Display;
import android.view.Surface;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import com.google.ar.core.Frame;
import com.google.ar.core.Session;
import com.google.ar.core.TrackingState;

import java.util.List;

import ru.nstu.navigator_arcore.Model;
import ru.nstu.navigator_arcore.tools.BoundingBox;
import ru.nstu.navigator_arcore.tools.OverlayView;


public class ARCoreRenderer implements GLSurfaceView.Renderer {

    private Session session;
    private BackgroundRenderer backgroundRenderer;

    private Image latestImage = null;
    private boolean isProcessing = false;
    private Model YOLOModel;
    private List<BoundingBox> lastBoxes;
    private OverlayView overlayView;

    public ARCoreRenderer(Session session, Model model,OverlayView overlayView, Context context) {
        this.session = session;
        this.backgroundRenderer = new BackgroundRenderer();
        this.context = context;
        this.YOLOModel = model;
        this.overlayView = overlayView;
    }

    private Context context;

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        backgroundRenderer.createOnGlThread(context);
        session.setCameraTextureName(backgroundRenderer.getTextureId());
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        GLES20.glViewport(0, 0, width, height);
        if (session != null) {
            session.setDisplayGeometry(
                    Surface.ROTATION_0,
                    width,
                    height
            );
        }
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        if (session == null) return;

        GLES20.glClearColor(0f, 0f, 0f, 1f);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);

        try {
            Frame frame = session.update();
            if (frame.getCamera().getTrackingState() != TrackingState.TRACKING) return;
            backgroundRenderer.draw(frame);

            if (!isProcessing) {
                try {
                    latestImage = frame.acquireCameraImage();
                    if (latestImage != null) {
                        isProcessing = true;
                        new Thread(() -> {
                            List<BoundingBox> boxes = YOLOModel.analyzeImage(latestImage, 0);
                            latestImage.close();
                            latestImage = null;

                            overlayView.post(() -> overlayView.setResults(boxes, YOLOModel.Classes));

                            isProcessing = false;
                        }).start();
                    }
                } catch (Exception e) {
                    Log.e("Error LOG.E", "ARCoreRenderer (onDrawFrame): "+e.getMessage());
                }
            }

        } catch (Exception e) {
            Log.e("Error LOG.E", "ARCoreRenderer (onDrawFrame): "+e.getMessage());
        }
    }
}
