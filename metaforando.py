import cv2
import os
import tkinter as tk
from tkinter import filedialog, ttk
from deepface import DeepFace
import shutil
from collections import Counter
import threading
import time

# Função para desenhar texto com fundo
def draw_text_with_background_opencv(img, text, face_num, fontScale=1, thickness=2, color=(255, 255, 255), bg=(0, 0, 0)):
    padding = 10
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)

    # Limita o tamanho do texto ao tamanho da imagem
    max_width = img.shape[1] - 2 * padding
    if text_width > max_width:
        fontScale = fontScale * max_width / text_width
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)

    y = img.shape[0] - padding - face_num * (text_height + baseline + padding)
    x = padding

    cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y), bg, thickness=cv2.FILLED)

    cv2.putText(img, text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)

    return img

# Função para extrair quadros do vídeo
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not os.path.exists('frames'):
        os.makedirs('frames')

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # Salve o quadro como uma imagem
            cv2.imwrite(f'frames/frame{frame_count}.jpg', frame)
            frame_count += 1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return frame_count

# Função para analisar os frames
def analyze_frames(video_path, output_path, fps, progress_bar_extraction, progress_bar_analysis, result_label):
    frame_count = extract_frames(video_path)  # Extrai os quadros do vídeo

    emotions = []
    prev_emotion = None

    for i in range(frame_count):
        img_path = f'frames/frame{i}.jpg'
        img = cv2.imread(img_path)

        results = DeepFace.analyze(img, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)

        for j, result in enumerate(results):
            text = f"Face {j + 1}:\nAge: {result['age']}\nGender: {result['gender']}\nRace: {result['dominant_race']}\nEmotion: {result['dominant_emotion']}"
            emotions.append(result['dominant_emotion'])
            img = draw_text_with_background_opencv(img, text, j)

            if prev_emotion:
                emotions.extend([prev_emotion] * (30 // fps - j))

        cv2.imwrite(f'analyzed_frames/frame{i}.jpg', img)

        prev_emotion = result['dominant_emotion']

        # Atualize a barra de progresso de análise de frames
        progress = min(int((i / frame_count) * 100), 100)
        progress_bar_analysis["value"] = progress

        # Calcule o tempo estimado restante
        current_time = time.time()
        elapsed_time = current_time - start_time
        estimated_time_remaining = elapsed_time / (i + 1) * (frame_count - i - 1)
        hours, remainder = divmod(estimated_time_remaining, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_remaining_str = f"Tempo estimado restante: {int(hours)}h {int(minutes)}m {int(seconds)}s"
        result_label.config(text=f"Análise em andamento... Progresso: {progress}%\n{time_remaining_str}")
        root.update()

    img_array = []
    for i in range(frame_count):
        img_path = f'analyzed_frames/frame{i}.jpg'
        img = cv2.imread(img_path)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()

    shutil.rmtree('frames')
    shutil.rmtree('analyzed_frames')

    emotion_counter = Counter(emotions)
    total_emotions = sum(emotion_counter.values())
    emotion_percentages = {emotion: count / total_emotions * 100 for emotion, count in emotion_counter.items()}

    return emotion_percentages

# Função para selecionar o vídeo e iniciar a análise
def select_and_analyze_video():
    global start_time  # Declare a variável global para rastrear o tempo de início
    start_time = time.time()  # Registre o tempo de início da análise

    # Abra a caixa de diálogo para selecionar o vídeo
    filepath = filedialog.askopenfilename()

    # Solicite ao usuário a taxa de quadros por segundo (FPS) para análise
    fps = int(input("Digite a taxa de quadros por segundo para análise: "))

    # Configure a barra de progresso para extração de frames
    progress_frame_extraction["maximum"] = 100
    progress_frame_extraction["value"] = 0

    # Configure a barra de progresso para análise de frames
    progress_frame_analysis["maximum"] = 100
    progress_frame_analysis["value"] = 0

    # Extraia e analise os frames em segundo plano
    def analyze_in_background():
        result_label.config(text="Análise em andamento...")
        root.update()

        emotion_percentages = analyze_frames(filepath, 'project.avi', fps, progress_frame_extraction, progress_frame_analysis, result_label)

        result = "Análise concluída. Resultados salvos em 'project.avi'.\n\nEmoções detectadas:\n"
        for emotion, percentage in emotion_percentages.items():
            result += f"{emotion}: {percentage:.2f}%\n"

        result_label.config(text=result)

    thread = threading.Thread(target=analyze_in_background)
    thread.start()

# Crie a janela principal
root = tk.Tk()
root.title("Análise de Vídeo")

# Crie um botão para selecionar o vídeo e iniciar a análise
analyze_button = tk.Button(root, text="Selecionar Vídeo e Iniciar Análise", command=select_and_analyze_video)
analyze_button.pack()

# Crie uma barra de progresso para extração de frames
progress_frame_extraction = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate")
progress_frame_extraction.pack()

# Crie uma barra de progresso para análise de frames
progress_frame_analysis = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate")
progress_frame_analysis.pack()

# Rótulo para exibir o resultado
result_label = tk.Label(root, text="")
result_label.pack()

# Inicie o loop principal
root.mainloop()
