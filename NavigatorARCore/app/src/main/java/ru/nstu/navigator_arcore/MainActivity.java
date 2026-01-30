package ru.nstu.navigator_arcore;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.util.Log;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Config;
import com.google.ar.core.Session;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import ru.nstu.navigator_arcore.renderer.ARCoreRenderer;
import ru.nstu.navigator_arcore.tools.OverlayView;

public class MainActivity extends AppCompatActivity {
    //---------------------------------------------LOGCAT
    private final String TAG = "LOG.E";
    private final String START_MESSAGE = "MainActivity";

    //---------------------------------------------Permissions
    private final int REQUEST_PERMISSION_CODE = 102030;
    private final String[] REQUIRED_PERMISSIONS = new String[] {"android.permission.CAMERA"};

    //---------------------------------------------Load File
    private static final int PICK_MODEL_FILE = 1001;
    private static final int PICK_CLASSES_FILE = 1002;
    private Uri modelUri;
    private Uri classesUri;

    //---------------------------------------------ARCore
    private Session mSession;
    private ARCoreRenderer arcRenderer;
    private boolean installARCoreRequested = true;

    //---------------------------------------------Model
    private Model YOLOModel;
    private final String assetsFileModel = "best.pte";
    private final String assetsFileClasses = "classes.txt";
    private File tempModelFile;
    private File tempClassesFile;

    //---------------------------------------------UI
    private OverlayView overlayView;
    private GLSurfaceView glSurfaceView;
    private TextView reqText;
    private TextView loadModelText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        this.glSurfaceView = findViewById(R.id.glSurfaceView);
        this.overlayView   = findViewById(R.id.overlay);
        this.reqText       = findViewById(R.id.reqText);
        this.loadModelText = findViewById(R.id.loadModelText);

        this.arcRenderer = new ARCoreRenderer(this.overlayView, this);
        setupGlSurfaceView();

        findViewById(R.id.btnUseAssetsFile).setOnClickListener(event->{
            try{
                this.YOLOModel = new Model(this.assetsFileModel,this.assetsFileClasses, this, true);
                this.arcRenderer.setModel(this.YOLOModel);
                this.loadModelText.setText("Загруженно из ASSETS");
            }catch (Exception e){
                Log.e(this.TAG, this.START_MESSAGE + " (onCreate): ", e);
            }
        });
        findViewById(R.id.btnPickModel).setOnClickListener(event->{
            pickModelLauncher.launch(createPickModelIntent());
        });
    }

    //---------------------------------------------Loading from external storage
    private final ActivityResultLauncher<Intent> pickClassesLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
                    result -> {
                        if (result.getResultCode() == Activity.RESULT_OK && result.getData() != null) {
                            classesUri = result.getData().getData();
                            loadExternalModel();
                        }
                    });
    private final ActivityResultLauncher<Intent> pickModelLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
                    result -> {
                        if (result.getResultCode() == Activity.RESULT_OK && result.getData() != null) {
                            modelUri = result.getData().getData();
                            pickClassesLauncher.launch(createPickClassesIntent());
                        }
                    });

    private Intent createPickModelIntent() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        return intent;
    }
    private Intent createPickClassesIntent() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("text/plain");
        return intent;
    }

    private File copyUriToCache(Uri uri, String fileName) throws IOException {
        File outFile = new File(getCacheDir(), fileName);
        try (InputStream in = getContentResolver().openInputStream(uri);
             FileOutputStream out = new FileOutputStream(outFile)) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
        }
        return outFile;
    }
    private void loadExternalModel() {
        if (modelUri != null && classesUri != null) {
            try {
                // Copy to cache
                tempModelFile = copyUriToCache(modelUri, "tempModel.torchscript");
                tempClassesFile = copyUriToCache(classesUri, "tempClasses.txt");

                // Load PyTorch model
                YOLOModel = new Model(tempModelFile.getAbsolutePath(), tempClassesFile.getAbsolutePath(),this,false);

                if (arcRenderer != null) arcRenderer.setModel(YOLOModel);

                loadModelText.setText("Загружено: " + getFileNameFromUri(modelUri));

            } catch (Exception e) {
                Log.e(TAG, "Error loading external model", e);
            }
        }
    }
    private void deleteTempFile(File file) {
        if (file != null && file.exists()) {
            boolean deleted = file.delete();
            if (!deleted) Log.w(TAG, "Failed to delete temp file: " + file.getAbsolutePath());
        }
    }
    private String getFileNameFromUri(Uri uri) {
        String result = null;
        if ("content".equals(uri.getScheme())) {
            try (android.database.Cursor cursor = getContentResolver().query(uri, null, null, null, null)) {
                if (cursor != null && cursor.moveToFirst()) {
                    int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                    if (index != -1) {
                        result = cursor.getString(index);
                    }
                }
            } catch (Exception e) {
                Log.w(TAG, START_MESSAGE + " Failed to get file name from uri", e);
            }
        }

        if (result == null) {
            result = uri.getLastPathSegment();
        }
        return result;
    }
    //---------------------------------------------

    private boolean _checkPermission(){
        for(String perm : REQUIRED_PERMISSIONS){
            if(ContextCompat.checkSelfPermission(this, perm) != PackageManager.PERMISSION_GRANTED) return false;
        }
        return true;
    }

    private void setupGlSurfaceView() {
        glSurfaceView.setEGLContextClientVersion(2);
        glSurfaceView.setPreserveEGLContextOnPause(true);
        glSurfaceView.setRenderer(this.arcRenderer);
        glSurfaceView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
    }

    @Override
    protected void onResume(){
        super.onResume();

        if (mSession == null) {
            Exception exception = null;
            String message = null;
            try {
                ArCoreApk.Availability availability = ArCoreApk.getInstance().checkAvailability(this);
                if (availability != ArCoreApk.Availability.SUPPORTED_INSTALLED) {
                    switch (ArCoreApk.getInstance().requestInstall(this, !installARCoreRequested)) {
                        case INSTALL_REQUESTED:
                            installARCoreRequested = false;
                            return;
                        case INSTALLED:
                            break;
                    }
                }

                if (!_checkPermission()) {
                    ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_PERMISSION_CODE);
                    return;
                }

                mSession = new Session(this);
                this.arcRenderer.setSession(mSession);
            } catch (UnavailableArcoreNotInstalledException | UnavailableUserDeclinedInstallationException e) {
                message = "Please install ARCore";
                exception = e;
            } catch (UnavailableApkTooOldException e) {
                message = "Please update ARCore";
                exception = e;
            } catch (UnavailableDeviceNotCompatibleException e) {
                message = "This device does not support AR";
                exception = e;
            } catch (Exception e) {
                message = "Failed to create AR session";
                exception = e;
            }

            if (message != null) {
                reqText.setText(message);
                Log.e(TAG, START_MESSAGE + " Exception creating session", exception);
                return;
            }
        }

        try {
            Config config = new Config(mSession);
            if (mSession.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                config.setDepthMode(Config.DepthMode.AUTOMATIC);
            } else {
                config.setDepthMode(Config.DepthMode.DISABLED);
            }
            mSession.configure(config);
            mSession.resume();
        } catch (CameraNotAvailableException e) {
            reqText.setText("Camera not available. Try restarting the app.");
            mSession = null;
            return;
        }

        glSurfaceView.onResume();
    }

    @Override
    protected void onDestroy() {
        deleteTempFile(tempModelFile);
        deleteTempFile(tempClassesFile);

        if(mSession != null){
           mSession.close();
           mSession = null;
        }

        super.onDestroy();
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mSession != null) {
            glSurfaceView.onPause();
            mSession.pause();
        }
    }
}