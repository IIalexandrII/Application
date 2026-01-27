package ru.nstu.navigator_arcore;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Config;
import com.google.ar.core.Session;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import ru.nstu.navigator_arcore.renderer.ARCoreRenderer;
import ru.nstu.navigator_arcore.tools.OverlayView;


public class MainActivity extends AppCompatActivity {
    private final int REQUEST_PERMISSION_CODE = 102030;
    private final String[] REQUIRED_PERMISSIONS = new String[] {"android.permission.CAMERA"};

    private TextView reqText;
    private GLSurfaceView glSurfaceView;
    private Session mSession = null;
    private boolean mUserRequestedInstall = true;

    private Model YOLOModel;
    private OverlayView overlayView;

    //--------------------------------------------------------
    private static final int PICK_MODEL_FILE = 1001;
    private static final int PICK_CLASSES_FILE = 1002;
    private Uri modelUri;
    private Uri classesUri;

    private boolean rendererInitialized = false;
    //--------------------------------------------------------


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
        this.reqText = findViewById(R.id.reqText);
        this.overlayView = findViewById(R.id.overlay);
        this._checkARSupport();
        findViewById(R.id.btnPickModel).setOnClickListener(event->{
            pickModelFile();
        });
        this.glSurfaceView = findViewById(R.id.glSurfaceView);
//        try {
//            this.YOLOModel = new Model("best.torchscript","classes.txt", this);
//        }catch (Exception e){
//            Log.e("Error LOG.E", "MainActivity (onCreate): "+e.getMessage());
//        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if(!this._checkPermission()){
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_PERMISSION_CODE);
        }
        try {
            if (mSession == null) {
                ArCoreApk.InstallStatus installStatus = ArCoreApk.getInstance().requestInstall(this, mUserRequestedInstall);

                switch (installStatus) {
                    case INSTALLED:
                        mSession = new Session(this);
                        Config config = new Config(mSession);
                        mSession.configure(config);
//                        this._setupRenderer();
                        trySetupRenderer();
                        break;

                    case INSTALL_REQUESTED:
                        mUserRequestedInstall = false;
                        reqText.setText("Установка ARCore...");
                        reqText.setVisibility(TextView.VISIBLE);
                        break;
                }
            }
            if(mSession != null) mSession.resume();
            if (rendererInitialized) {
                glSurfaceView.onResume();
            }

        } catch (UnavailableUserDeclinedInstallationException e) {
            reqText.setText("Для работы приложения требуется ARCore");
            reqText.setVisibility(TextView.VISIBLE);
            Toast.makeText(this, "Установите Google Play Services for AR из Play Маркета", Toast.LENGTH_LONG).show();
        } catch (Exception e) {
            reqText.setText("Ошибка инициализации ARCore");
            reqText.setVisibility(TextView.VISIBLE);
            Log.e("Error LOG.E", "MainActivity (onResume): "+e.getMessage());
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(mSession != null) mSession.pause();
        if (rendererInitialized) glSurfaceView.onPause();

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(mSession != null) mSession.close();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == REQUEST_PERMISSION_CODE) {
            if (!this._checkPermission()) {
                reqText.setText("Для работы приложения требуется доступ к камере");
                Toast.makeText(this, "Разрешите доступ к камере в настройках", Toast.LENGTH_LONG).show();
            }
        }
    }

    private void _setupRenderer(){
        this.glSurfaceView.setEGLContextClientVersion(2);
        this.glSurfaceView.setPreserveEGLContextOnPause(true);
        this.glSurfaceView.setRenderer(new ARCoreRenderer(mSession, this.YOLOModel, this.overlayView, this));
        this.glSurfaceView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
    }
    private boolean _checkPermission(){
        for(String perm : REQUIRED_PERMISSIONS){
            if(ContextCompat.checkSelfPermission(this, perm) != PackageManager.PERMISSION_GRANTED) return false;
        }
        return true;
    }
    private boolean _checkARSupport(){
        ArCoreApk.Availability availability = ArCoreApk.getInstance().checkAvailability(this);
        if (availability.isUnsupported()){
            this.reqText.setText("ARCore не поддерживается на этом устройстве");
            return false;
        }
        this.reqText.setVisibility(TextView.INVISIBLE);
        return true;
    }

    private void pickModelFile() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        startActivityForResult(intent, PICK_MODEL_FILE);
    }

    private void pickClassesFile() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("text/plain");
        startActivityForResult(intent, PICK_CLASSES_FILE);
    }

    private File copyUriToInternal(Uri uri, String fileName) throws IOException {
        File outFile = new File(getFilesDir(), fileName);

        try (InputStream in = getContentResolver().openInputStream(uri);
             FileOutputStream out = new FileOutputStream(outFile)) {

            byte[] buffer = new byte[4096];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
        }
        return outFile;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode != Activity.RESULT_OK || data == null) return;

        if (requestCode == PICK_MODEL_FILE) {
            modelUri = data.getData();
            pickClassesFile();
            return;
        }

        if (requestCode == PICK_CLASSES_FILE) {
            classesUri = data.getData();
        }

        if (modelUri != null && classesUri != null) {
            try {
                File modelFile = copyUriToInternal(modelUri, "model.torchscript");
                File classesFile = copyUriToInternal(classesUri, "classes.txt");

                YOLOModel = new Model(
                        modelFile.getAbsolutePath(),
                        classesFile.getAbsolutePath(),
                        this
                );

                trySetupRenderer();

            } catch (Exception e) {
                Log.e("Error LOG.E", "MainActivity (onActivityResult)" + e.getMessage());
            }
        }
    }

    private void trySetupRenderer() {
        if (rendererInitialized) return;

        if (mSession == null) {
            Log.d("Renderer", "Session not ready");
            return;
        }

        if (YOLOModel == null) {
            Log.d("Renderer", "YOLOModel not ready");
            return;
        }

        if (glSurfaceView == null) {
            Log.d("Renderer", "GLSurfaceView not ready");
            return;
        }

        _setupRenderer();
        rendererInitialized = true;

        Log.d("Renderer", "Renderer successfully initialized");
    }
}