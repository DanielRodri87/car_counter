import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Inicializa a lista de rastreadores e contadores de carros
trackers = cv2.MultiTracker_create()
car_count = 0

def select_video_file():
    # Abre o explorador de arquivos para selecionar um vídeo
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    return file_path

def process_video(video_path):
    global car_count

    # Carrega o modelo MobileNet SSD e os arquivos de configuração
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

    # Abre o vídeo
    cap = cv2.VideoCapture(video_path)

    # Verifica se o vídeo foi aberto com sucesso
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Atualiza os rastreadores existentes
        success, boxes = trackers.update(frame)

        # Desenha retângulos ao redor dos objetos rastreados
        for i, box in enumerate(boxes):
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Prepara o frame para detecção
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # Faz a detecção de objetos
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.4:  # Ajusta o nível de confiança
                idx = int(detections[0, 0, i, 1])

                if idx == 7:  # Classe 7 é 'carro'
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    new_box = (startX, startY, endX-startX, endY-startY)

                    # Verifica se a caixa detectada se sobrepõe a uma já rastreada
                    if not is_box_already_tracked(new_box, boxes):
                        # Adiciona novo rastreador
                        tracker = cv2.TrackerKCF_create()
                        trackers.add(tracker, frame, new_box)
                        car_count += 1

        # Exibe o contador de carros
        cv2.putText(frame, f"Carros: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Mostra o vídeo com as marcações
        cv2.imshow('Video', frame)

        # Sai se 'q' for pressionado
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def is_box_already_tracked(new_box, tracked_boxes):
    for box in tracked_boxes:
        iou = compute_iou(new_box, box)
        if iou > 0.3: 
            return True
    return False

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def main():
    root = tk.Tk()
    root.withdraw() 

    video_path = select_video_file()

    if video_path:
        process_video(video_path)
    else:
        print("Nenhum vídeo selecionado.")

if __name__ == "__main__":
    main()
