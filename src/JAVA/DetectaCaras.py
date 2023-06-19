import cv2
import face_recognition

# Cargar la imagen de referencia
reference_image = face_recognition.load_image_file("referencia.jpg")
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Inicializar la cámara
video_capture = cv2.VideoCapture(0)

while True:
    # Capturar un fotograma de la cámara
    ret, frame = video_capture.read()

    # Convertir el fotograma a RGB (necesario para face_recognition)
    rgb_frame = frame[:, :, ::-1]

    # Detectar rostros en el fotograma
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Comparar cada rostro detectado con la imagen de referencia
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([reference_encoding], face_encoding)

        if True in matches:
            # Si hay una coincidencia, dibujar un cuadro alrededor del rostro
            top, right, bottom, left = face_locations[matches.index(True)]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Mostrar el fotograma resultante
    cv2.imshow('Video', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
video_capture.release()
cv2.destroyAllWindows()
