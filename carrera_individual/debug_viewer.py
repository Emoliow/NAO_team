"""
Muestra la camara anotada del NAO en tiempo real.

Uso:
  1. En el NAO:    python carrera.py
  2. En la laptop: python3 debug_viewer.py
"""

import socket
import struct
import numpy as np
import cv2
import sys
import time

NAO_IP   = "192.168.0.30"   # <-- IP del NAO
DEBUG_PORT = 5000


def recvall(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(min(n - len(data), 65536))
        if not chunk:
            return None
        data += chunk
    return data


def main():
    ip = NAO_IP
    if len(sys.argv) > 1:
        ip = sys.argv[1]

    print("Conectando a %s:%d ..." % (ip, DEBUG_PORT))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)

    try:
        sock.connect((ip, DEBUG_PORT))
    except socket.error as e:
        print("No se pudo conectar: %s" % e)
        print("Asegurate de que carrera.py este corriendo en el NAO.")
        return

    sock.settimeout(5)
    print("Conectado. Presiona 'q' para salir.")

    hdr_size = struct.calcsize("III")
    fps_time = time.time()
    fps = 0.0

    try:
        while True:
            # Recibir header: width, height, jpeg_size
            hdr = recvall(sock, hdr_size)
            if hdr is None:
                print("Conexion cerrada.")
                break

            w, h, jpg_size = struct.unpack("III", hdr)

            # Recibir JPEG comprimido
            jpg_data = recvall(sock, jpg_size)
            if jpg_data is None:
                print("Conexion cerrada.")
                break

            # Decodificar JPEG
            jpg_arr = np.frombuffer(jpg_data, dtype=np.uint8)
            frame = cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Escalar para que se vea bien en la laptop
            display = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)

            # FPS
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (now - fps_time + 1e-9))
            fps_time = now
            cv2.putText(display, "FPS: %.1f" % fps, (540, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("NAO Debug Viewer", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except socket.timeout:
        print("Timeout: no llegan frames.")
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        cv2.destroyAllWindows()
        print("Viewer cerrado.")


if __name__ == "__main__":
    main()
