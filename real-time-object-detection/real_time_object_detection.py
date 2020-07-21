from imutils.video import VideoStream,FPS
import numpy as np
import argparse
import imutils
import time
import cv2
#Lista de etiquetas de clase MobileNet SSD fue entrenado para detectar
CLASSES = ["Fondo", "Avion", "Bicicleta", "Ave", "Bote", "Botella", "Autobus", "Carro", "Gato", "Silla", "Vaca", "Comedor", "Perro", "Casa", "Motocicleta", "persona", "maceta de planta", "Oveja", "Mueble", "Tren", "Pantalla"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
print("[INFO] Cargando modelo")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
print("[INFO] inicio de transmisión de video...")
vs = VideoStream(0).start()
time.sleep(2.0)
fps = FPS().start()#Inicialice el contador de FPS

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=1250)

	#toma las dimensiones del marco y conviértelo en un blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (370, 370)),0.007843, (300, 300), 127.5)
	#pasar el blob a través de la red y obtener las detecciones y predicciones
	net.setInput(blob)
	detections = net.forward()
	#Recorrer las detecciones
	for i in np.arange(0, detections.shape[2]):
		#extraer la confianza (i.e., la probabilidad) asociada con la predicción
		confidence = detections[0, 0, i, 2]

		#filtrar las detecciones débiles asegurando que la "confianza" es mayor que la confianza mínima
		if confidence > 0.2:
			#extraiga el índice de la etiqueta de clase de las `detecciones`, luego calcule las coordenadas (x, y) del cuadro delimitador para el objeto
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			#dibujar la predicción en el marco
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			
	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) & 0xFF == ord("q"):#Salir
		break
	fps.update()#Actualizar el contador de FPS
fps.stop()# detener el temporizador y mostrar información FPS
print("[INFO] tiempo transcurrido: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))