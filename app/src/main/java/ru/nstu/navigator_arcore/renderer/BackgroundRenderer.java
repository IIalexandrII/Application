package ru.nstu.navigator_arcore.renderer;

import android.content.Context;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;

import com.google.ar.core.Coordinates2d;
import com.google.ar.core.Frame;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class BackgroundRenderer {
    private static final float[] QUAD_COORDS = {
            -1f, -1f,
            1f, -1f,
            -1f,  1f,
            1f,  1f
    };

    private final FloatBuffer quadCoords = ByteBuffer.allocateDirect(QUAD_COORDS.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();

    private final FloatBuffer quadTexCoords = ByteBuffer.allocateDirect(QUAD_COORDS.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();

    private int textureId = -1;
    private int program;

    private int aPosition;
    private int aTexCoord;
    private int uTexture;

    public int getTextureId() {
        return textureId;
    }

    public void createOnGlThread(Context context) {
        quadCoords.put(QUAD_COORDS).position(0);

        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        textureId = textures[0];

        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);

        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        String vertexShader =
                "attribute vec4 a_Position;\n" +
                        "attribute vec2 a_TexCoord;\n" +
                        "varying vec2 v_TexCoord;\n" +
                        "void main() {\n" +
                        "  gl_Position = a_Position;\n" +
                        "  v_TexCoord = a_TexCoord;\n" +
                        "}";

        String fragmentShader =
                "#extension GL_OES_EGL_image_external : require\n" +
                        "precision mediump float;\n" +
                        "uniform samplerExternalOES u_Texture;\n" +
                        "varying vec2 v_TexCoord;\n" +
                        "void main() {\n" +
                        "  gl_FragColor = texture2D(u_Texture, v_TexCoord);\n" +
                        "}";

        // Собираю программу из шейдеров.
        program = createProgram(vertexShader, fragmentShader);

        // Запоминаю “адреса” переменных в шейдерах.
        aPosition = GLES20.glGetAttribLocation(program, "a_Position");
        aTexCoord = GLES20.glGetAttribLocation(program, "a_TexCoord");
        uTexture = GLES20.glGetUniformLocation(program, "u_Texture");
    }

    public void draw(Frame frame) {
        // Если текстура не создана — рисовать нечего.
        if (textureId == -1) return;

        // ARCore может по-разному выдавать картинку камеры (ориентация/кроп),
        // поэтому UV пересчитываю каждый кадр из координат квадрата.
        float[] transformed = new float[QUAD_COORDS.length];
        frame.transformCoordinates2d(
                Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
                QUAD_COORDS,
                Coordinates2d.TEXTURE_NORMALIZED,
                transformed
        );

        // Обновляю UV-буфер свежими координатами.
        quadTexCoords.position(0);
        quadTexCoords.put(transformed);
        quadTexCoords.position(0);

        // Фон — это просто плоский “экран”, глубина тут только мешает.
        GLES20.glDisable(GLES20.GL_DEPTH_TEST);
        GLES20.glUseProgram(program);

        // Подключаю текстуру камеры в 0-й слот и говорю шейдеру, откуда читать.
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);
        GLES20.glUniform1i(uTexture, 0);

        // Атрибут позиций вершин квадрата.
        quadCoords.position(0);
        GLES20.glVertexAttribPointer(aPosition, 2, GLES20.GL_FLOAT, false, 0, quadCoords);
        GLES20.glEnableVertexAttribArray(aPosition);

        // Атрибут UV.
        quadTexCoords.position(0);
        GLES20.glVertexAttribPointer(aTexCoord, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords);
        GLES20.glEnableVertexAttribArray(aTexCoord);

        // Рисуем 2 треугольника, которые покрывают весь экран.
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

        // Прибираюсь за собой (чтобы не ломать следующий рендер).
        GLES20.glDisableVertexAttribArray(aPosition);
        GLES20.glDisableVertexAttribArray(aTexCoord);

        // Возвращаю depth test — дальше могут рисоваться 3D-объекты.
        GLES20.glEnable(GLES20.GL_DEPTH_TEST);
    }

    // Компиляция одного шейдера (vertex/fragment).
    private static int loadShader(int type, String code) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, code);
        GLES20.glCompileShader(shader);
        return shader;
    }

    // Линкуем программу: vertex + fragment шейдеры вместе.
    private static int createProgram(String vtx, String frag) {
        int vs = loadShader(GLES20.GL_VERTEX_SHADER, vtx);
        int fs = loadShader(GLES20.GL_FRAGMENT_SHADER, frag);
        int prog = GLES20.glCreateProgram();
        GLES20.glAttachShader(prog, vs);
        GLES20.glAttachShader(prog, fs);
        GLES20.glLinkProgram(prog);
        return prog;
    }
}
