"""
CARRERA INDIVIDUAL - NAO V4
Corre DENTRO del NAO: python carrera.py
Optimizado para Intel Atom Z530.

Controles:
  - Boton frontal cabeza: iniciar/detener
  - El robot sigue el centro entre dos lineas negras
  - Se detiene automaticamente al detectar la meta roja

Debug (opcional):
  Corriendo debug_viewer.py
"""

import sys
import time
import socket
import struct
from naoqi import ALProxy

import cv2
import numpy as np

NAO_IP   = "127.0.0.1"   # localhost porque corre en el propio NAO
NAO_PORT = 9559

# -- Camara --
CAM_BOTTOM  = 1
RESOLUTION  = 1      # 320x240 (clave para el Atom)
COLORSPACE  = 11     # BGR
FPS         = 15
HEAD_PITCH  = -0.30   # inclinacion de cabeza (rad). Mas alto = ve mas cerca

# -- Vision --
THRESH      = 140     # umbral de blanco (ajustar segun iluminacion)
ROI_TOP     = 0.50   # fraccion: solo procesa desde aqui hacia abajo
MIN_PX      = 15     # pixeles minimos para considerar una linea valida
LANE_HALF_F = 0.28   # mitad del ancho del carril como fraccion de la imagen

# -- Meta roja --
META_THRESH        = 180   # umbral de blanco para la meta (ajustar segun iluminacion)
META_MIN_ROW_FILL  = 180   # pixeles blancos minimos en UNA fila para detectar meta
                           # con 320px de ancho, 180 = ~56% del ancho de la imagen
# -- PID --
KP = 0.005
KI = 0.0
KD = 0.002

# -- Movimiento --
FWD_SPEED   = 0.6    # velocidad de avance (0..1). Empieza bajo y sube.

FWD_START   = 0.8   # velocidad inicial al arrancar
RAMP_TIME   = 5.0    # segundos para llegar de FWD_START a FWD_SPEED

LOST_SPEED  = 0.15   # velocidad cuando pierde ambas lineas
CURVE_BRAKE = 0.5    # cuanto frena en curvas (0..1)

# -- Debug remoto --
DEBUG_ENABLED = True  # False para competencia (ahorra CPU)
DEBUG_PORT    = 5000



class PID(object):
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev = 0.0
        self.integ = 0.0

    def step(self, error):
        self.integ += error
        self.integ = max(-40000, min(40000, self.integ))
        deriv = error - self.prev
        self.prev = error
        return self.kp * error + self.ki * self.integ + self.kd * deriv

    def reset(self):
        self.prev = 0.0
        self.integ = 0.0


class DebugStream(object):
    """Envia frames anotados a la laptop. No bloquea si no hay viewer."""
    def __init__(self, port):
        self.conn = None
        self.server = None
        if not DEBUG_ENABLED:
            return
        try:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind(("0.0.0.0", port))
            self.server.listen(1)
            self.server.setblocking(False)
            print("[DEBUG] Esperando viewer en puerto %d..." % port)
        except Exception as e:
            print("[DEBUG] No se pudo iniciar: %s" % e)
            self.server = None

    def check_connection(self):
        if self.server is None:
            return
        if self.conn is not None:
            return
        try:
            self.conn, addr = self.server.accept()
            self.conn.settimeout(0.5)
            self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("[DEBUG] Viewer conectado desde %s" % str(addr))
        except socket.error:
            pass

    def send(self, frame):
        if self.conn is None:
            self.check_connection()
            return
        try:
            h, w = frame.shape[:2]
            # Comprimir a JPEG para reducir ancho de banda
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            jpg_bytes = jpg.tostring()
            header = struct.pack("III", w, h, len(jpg_bytes))
            self.conn.sendall(header + jpg_bytes)
        except socket.error:
            print("[DEBUG] Viewer desconectado.")
            self.conn = None

    def close(self):
        if self.conn:
            self.conn.close()
        if self.server:
            self.server.close()


def get_frame(video, client_id):
    """Captura un frame de la camara. Retorna numpy array o None."""
    img = video.getImageRemote(client_id)
    if img is None:
        return None
    w, h = img[0], img[1]
    data = bytes(bytearray(img[6]))
    frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
    return frame.copy()


def detectar_lineas(frame):
    """
    Detecta las lineas negras del carril.
    Optimizado: solo escala de grises + threshold + numpy.
    Sin contornos ni morfologia (pesados para el Atom).

    Retorna: (error, center_x, left_x, right_x, y_start)
    """
    h, w = frame.shape[:2]
    cx = w // 2
    y0 = int(h * ROI_TOP)

    # Solo la ROI inferior, solo escala de grises
    roi = frame[y0:, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Threshold: negro = linea
    mask = cv2.threshold(gray, THRESH, 255, cv2.THRESH_BINARY)[1]

    # Tomar solo una franja horizontal del medio de la ROI (mas rapido)
    rh = mask.shape[0]
    scan_y0 = rh // 3
    scan_y1 = 2 * rh // 3
    scan = mask[scan_y0:scan_y1, :]

    # Columnas donde hay negro
    cols = np.where(scan > 0)
    if cols[1].size < MIN_PX:
        return None, None, None, None, y0

    xs = cols[1]

    # Separar izquierda / derecha del centro
    left_xs  = xs[xs < cx]
    right_xs = xs[xs >= cx]

    lx = int(left_xs.mean())  if left_xs.size  > MIN_PX else None
    rx = int(right_xs.mean()) if right_xs.size > MIN_PX else None

    # Calcular centro del carril
    half = int(w * LANE_HALF_F)
    if   lx is not None and rx is not None:
        center = (lx + rx) // 2
    elif lx is not None:
        center = lx + half
    elif rx is not None:
        center = rx - half
    else:
        return None, None, None, None, y0

    error = center - cx
    return error, center, lx, rx, y0

def detectar_meta(frame):
    """
    Detecta la linea blanca horizontal de meta.
    Busca una franja con muchos pixeles blancos DISTRIBUIDOS
    horizontalmente en la parte inferior del frame.
    """
    h, w = frame.shape[:2]

    # Solo la mitad inferior (la meta estara cerca de los pies)
    roi = frame[h // 2:, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Pixeles blancos (por encima del umbral)
    _, mask = cv2.threshold(gray, META_THRESH, 255, cv2.THRESH_BINARY)

    # Contar pixeles blancos por fila
    # La meta es una linea HORIZONTAL: una fila tendra muchos pixeles blancos
    row_sums = np.sum(mask > 0, axis=1)  # pixeles blancos por fila

    # Si alguna fila tiene pixeles blancos a lo ancho de casi toda la imagen,
    # es la linea de meta (no una de las lineas de carril que son verticales)
    return bool(np.any(row_sums > META_MIN_ROW_FILL))

def anotar_debug(frame, error, center, lx, rx, y0, state, vx, wz, meta):
    """Dibuja anotaciones sobre el frame para debug."""
    h, w = frame.shape[:2]
def anotar_debug(frame, error, center, lx, rx, y0, state, vx, wz, meta):
    """Dibuja anotaciones sobre el frame para debug."""
    h, w = frame.shape[:2]
    cx = w // 2

    # Linea central (verde)
    cv2.line(frame, (cx, 0), (cx, h), (0, 255, 0), 1)

    # Lineas detectadas
    if lx is not None:
        cv2.line(frame, (lx, y0), (lx, h), (255, 0, 0), 2)
    if rx is not None:
        cv2.line(frame, (rx, y0), (rx, h), (255, 255, 0), 2)
    # Centro calculado del carril (rojo)
    if center is not None:
        cv2.line(frame, (center, y0), (center, h), (0, 0, 255), 2)

    # ROI
    cv2.line(frame, (0, y0), (w, y0), (100, 100, 100), 1)

    # Texto
    estado = "RUN" if state else "STOP"
    err_str = str(error) if error is not None else "---"
    txt = "%s e=%s wz=%.2f vx=%.2f" % (estado, err_str, wz, vx)
    cv2.putText(frame, txt, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    if meta:
        cv2.putText(frame, "META", (cx - 20, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame


def main():
    # ── Conectar modulos NAOqi (local) ──
    motion  = ALProxy("ALMotion",       NAO_IP, NAO_PORT)
    posture = ALProxy("ALRobotPosture", NAO_IP, NAO_PORT)
    video   = ALProxy("ALVideoDevice",  NAO_IP, NAO_PORT)
    memory  = ALProxy("ALMemory",       NAO_IP, NAO_PORT)

    # ── Postura inicial ──
    motion.wakeUp()
    posture.goToPosture("StandInit", 0.7)
    motion.setStiffnesses("Head", 1.0)
    motion.setAngles("HeadYaw",   0.0,        0.3)
    motion.setAngles("HeadPitch", HEAD_PITCH,  0.3)
    motion.moveInit()

    # ── Camara ──
    client_id = video.subscribeCamera("carrera", CAM_BOTTOM, RESOLUTION, COLORSPACE, FPS)

    # ── Debug stream (opcional) ──
    debug = DebugStream(DEBUG_PORT)

    # ── PID ──
    pid = PID(KP, KI, KD)

    # ── Estado ──
    state    = 0     # 0=STOP, 1=RUN
    prev_btn = 0.0
    vx = wz  = 0.0
    meta_reached = False
    current_speed = FWD_START   
    ramp_start    = 0.0    

    print("=== CARRERA INDIVIDUAL ===")
    print("Presiona el boton FRONTAL de la cabeza para iniciar.")
    print("Presionalo de nuevo para detener.")

    try:
        while True:
            # ── Boton frontal: toggle ──
            btn = memory.getData("FrontTactilTouched")
            if btn == 1.0 and prev_btn == 0.0:
                state = 1 - state
                if state == 1:
                    current_speed = FWD_START   # <-- agrega
                    ramp_start    = time.time() # <-- agrega
                    print(">> EN MARCHA")
                if state == 0:
                    motion.stopMove()
                    pid.reset()
                    vx = wz = 0.0
                    meta_reached = False
                print(">> %s" % ("EN MARCHA" if state else "DETENIDO"))
                time.sleep(0.3)
            prev_btn = btn

            # ── Capturar frame ──
            frame = get_frame(video, client_id)
            if frame is None:
                continue

            # ── Procesar vision ──
            error, center, lx, rx, y0 = detectar_lineas(frame)
            meta = detectar_meta(frame)

            # ── Decidir movimiento ──
            if state == 1:
                # Rampa de velocidad
                elapsed = time.time() - ramp_start
                if elapsed < RAMP_TIME:
                    t = elapsed / RAMP_TIME          # 0.0 -> 1.0
                    current_speed = FWD_START + (FWD_SPEED - FWD_START) * t
                else:
                    current_speed = FWD_SPEED

                if meta and not meta_reached:
                    meta_reached = True
                    pid.reset()
                    print(">> META DETECTADA - recto hasta boton")

                if meta_reached:
                    vx = current_speed
                    wz = 0.0
                    motion.moveToward(vx, 0.0, 0.0)
                elif error is not None:
                    # Seguir el carril
                    corr = pid.step(error)
                    corr = max(-1.0, min(1.0, corr))
                    wz = -corr
                    vx = current_speed * (1.0 - CURVE_BRAKE * min(1.0, abs(corr)))
                    motion.moveToward(vx, 0.0, wz)
                else:
                    # Perdio las lineas: avanzar lento sin girar
                    vx = min(LOST_SPEED, current_speed)
                    wz = 0.0
                    motion.moveToward(vx, 0.0, 0.0)

            # ── Debug stream (no bloquea si no hay viewer) ──
            if DEBUG_ENABLED:
                dbg = anotar_debug(frame, error, center, lx, rx, y0,
                                   state, vx, wz, meta)
                debug.send(dbg)

    except KeyboardInterrupt:
        print("\nInterrumpido por teclado.")
    finally:
        motion.stopMove()
        motion.rest()
        video.unsubscribe(client_id)
        debug.close()
        print("Terminado.")


if __name__ == "__main__":
    main()
