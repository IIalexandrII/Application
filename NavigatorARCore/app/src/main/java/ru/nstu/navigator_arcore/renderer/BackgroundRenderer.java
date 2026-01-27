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

    // Координаты прямоугольника на весь экран в NDC (-1..1).
    // Рисуем его через TRIANGLE_STRIP: 4 точки = 2 треугольника.
    private static final float[] QUAD_COORDS = {
            -1f, -1f,
            1f, -1f,
            -1f,  1f,
            1f,  1f
    };

    // Буфер под координаты квадрата (позиции вершин).
    private final FloatBuffer quadCoords =
            ByteBuffer.allocateDirect(QUAD_COORDS.length * 4)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer();

    // Буфер под UV (текстурные координаты).
    // Их каждый кадр дает ARCore (они могут меняться из-за поворота/кропа камеры).
    private final FloatBuffer quadTexCoords =
            ByteBuffer.allocateDirect(QUAD_COORDS.length * 4)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer();

    // ID текстуры, куда ARCore “подкладывает” изображение камеры (OES external texture).
    private int textureId = -1;
    private int program;

    // Локации атрибутов/юниформа в шейдерах (чтобы не искать каждый кадр).
    private int aPosition;
    private int aTexCoord;
    private int uTexture;

    public int getTextureId() {
        return textureId;
    }

    public void createOnGlThread(Context context) {
        // Один раз заливаю координаты квадрата в буфер.
        quadCoords.put(QUAD_COORDS).position(0);

        // Создаю внешнюю OES текстуру для камеры.
        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        textureId = textures[0];

        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);

        // Обычные параметры фильтрации/обертки — чтобы картинка не мылась и не “зацикливалась”.
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        // Вершинный шейдер: просто прокидывает позицию и UV дальше во фрагментный.
        String vertexShader =
                "attribute vec4 a_Position;\n" +
                        "attribute vec2 a_TexCoord;\n" +
                        "varying vec2 v_TexCoord;\n" +
                        "void main() {\n" +
                        "  gl_Position = a_Position;\n" +
                        "  v_TexCoord = a_TexCoord;\n" +
                        "}";

        // Фрагментный шейдер: читаем цвет из камеры (samplerExternalOES).
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
        if (textureId == -1) return;

        float[] transformed = new float[QUAD_COORDS.length];
        frame.transformCoordinates2d(
                Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
                QUAD_COORDS,
                Coordinates2d.TEXTURE_NORMALIZED,
                transformed
        );

        quadTexCoords.position(0);
        quadTexCoords.put(transformed);
        quadTexCoords.position(0);

        GLES20.glDisable(GLES20.GL_DEPTH_TEST);
        GLES20.glUseProgram(program);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);
        GLES20.glUniform1i(uTexture, 0);

        quadCoords.position(0);
        GLES20.glVertexAttribPointer(aPosition, 2, GLES20.GL_FLOAT, false, 0, quadCoords);
        GLES20.glEnableVertexAttribArray(aPosition);

        quadTexCoords.position(0);
        GLES20.glVertexAttribPointer(aTexCoord, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords);
        GLES20.glEnableVertexAttribArray(aTexCoord);

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

        GLES20.glDisableVertexAttribArray(aPosition);
        GLES20.glDisableVertexAttribArray(aTexCoord);

        GLES20.glEnable(GLES20.GL_DEPTH_TEST);
    }

    private static int loadShader(int type, String code) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, code);
        GLES20.glCompileShader(shader);
        return shader;
    }

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
