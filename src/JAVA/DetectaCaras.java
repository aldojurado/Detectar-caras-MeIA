import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_objdetect.*;

public class DetectaCaras {
    public static void main(String[] args) throws Exception {
        // Inicializar el capturador de video
        OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(0);
        grabber.start();

        // Cargar el clasificador de detección de rostros
        CascadeClassifier classifier = new CascadeClassifier();
        classifier.load("haarcascade_frontalface_default.xml");

        // Crear una ventana para mostrar el video en vivo
        CanvasFrame frame = new CanvasFrame("Face Detection");
        frame.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);

        // Bucle principal
        while (frame.isVisible()) {
            // Capturar el siguiente fotograma del video
            Frame videoFrame = grabber.grab();

            // Convertir el fotograma en una matriz de píxeles
            opencv_core.Mat mat = new opencv_core.Mat(videoFrame);
            opencv_core.Mat grayMat = new opencv_core.Mat();

            // Convertir la matriz a escala de grises
            opencv_imgproc.cvtColor(mat, grayMat, opencv_imgproc.CV_BGR2GRAY);

            // Detectar rostros en la matriz de escala de grises
            RectVector faces = new RectVector();
            classifier.detectMultiScale(grayMat, faces);

            // Dibujar rectángulos alrededor de los rostros detectados
            for (int i = 0; i < faces.size(); i++) {
                Rect face = faces.get(i);
                opencv_imgproc.rectangle(mat, face, new Scalar(0, 255, 0, 1));
            }

            // Mostrar el fotograma con los rectángulos de los rostros
            frame.showImage(videoFrame);
        }

        // Detener el capturador de video y cerrar la ventana
        grabber.stop();
        frame.dispose();
    }
}