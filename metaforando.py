
import os
import shutil
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import cv2
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Função para baixar o modelo pré-treinado
def download_model():
    url = "https://github.com/priya-dwivedi/face_and_emotion_detection/raw/master/emotion_detector_models/model_v6_23.hdf5"
    model_path = "model_v6_23.hdf5"
    r = requests.get(url, allow_redirects=True)
    with open(model_path, "wb") as f:
        f.write(r.content)
    print("Modelo baixado.")

# Função para carregar o modelo
def load_pretrained_model():
    model = load_model("model_v6_23.hdf5")
    print("Modelo carregado.")
    return model

# Função para verificar disponibilidade da GPU
def check_gpu():
    gpu_available = tf.config.list_physical_devices('GPU')
    if not gpu_available:
        print("GPU não está disponível. O processo será executado na CPU.")
    else:
        print("GPU está disponível. O processo será executado na GPU.")

# Função para abrir o vídeo para análise
def open_video():
    global cap
    file_path = filedialog.askopenfilename()
    cap = cv2.VideoCapture(file_path)
    print(f"Vídeo {file_path} aberto.")
    return cap

# Função para obter taxa de quadros do vídeo
def get_video_fps():
    global video_fps
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps_label.config(text=f"Taxa de quadros do vídeo: {video_fps}")
    return video_fps

# Função para coletar entrada do usuário para taxa de quadros
def get_user_fps():
    user_fps = fps_entry.get()
    return user_fps

# Função para o loop de análise de quadro
def analyze_frame(cap, model, analyze_fps, video_fps):
    frame_count = 0
    total_emotions = {}
    os.makedirs("frames", exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % (video_fps // int(analyze_fps)) == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray_frame[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                predictions = model.predict(roi)[0]
                label = ['Raiva', 'Desgosto', 'Medo', 'Felicidade', 'Tristeza', 'Surpresa', 'Neutralidade']
                emotion_probability = np.max(predictions)
                emotion = label[np.argmax(predictions)]

                total_emotions[emotion] = total_emotions.get(emotion, 0) + 1
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (0, 0, 255), -1)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            cv2.imwrite(f"frames/frame_{frame_count}.jpg", frame)

        frame_count += 1

    return total_emotions

# Função para salvar o vídeo analisado
def save_analyzed_video(cap):
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('final_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
    
    frame_files = [f for f in os.listdir("frames") if f.endswith(".jpg")]
    frame_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    for frame_file in frame_files:
        frame = cv2.imread(f"frames/{frame_file}")
        out.write(frame)
    
    out.release()
    print("Vídeo analisado salvo.")

# Função para mostrar estatísticas
def show_statistics(total_emotions):
    if total_emotions:
        print("Estatísticas das análises faciais:")
        for emotion, count in total_emotions.items():
            print(f"{emotion}: {count}")
    else:
        print("Nenhuma estatística de análise disponível. Verifique se a análise facial foi bem-sucedida.")

# Função principal para executar todas as tarefas
def main():
    global cap, video_fps
    open_video()
    get_video_fps()
    
def start_analysis():
    download_model()
    model = load_pretrained_model()
    check_gpu()
    analyze_fps = get_user_fps()
    total_emotions = analyze_frame(cap, model, analyze_fps, video_fps)
    save_analyzed_video(cap)
    show_statistics(total_emotions)
    shutil.rmtree("frames")  # Remove o diretório com os quadros extraídos
    print("Processo concluído.")

# Interface Tkinter
root = tk.Tk()
root.title("Análise de Emoção em Vídeo")

open_button = tk.Button(root, text="Abrir Vídeo", command=main)
open_button.pack()

fps_label = tk.Label(root, text="Taxa de quadros do vídeo: ")
fps_label.pack()
fps_entry = tk.Entry(root)
fps_entry.pack()

analyze_button = tk.Button(root, text="Iniciar Análise", command=start_analysis)
analyze_button.pack()

root.mainloop()
