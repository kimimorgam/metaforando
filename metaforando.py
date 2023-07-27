import cv2
import os
import tkinter as tk
from tkinter import filedialog
from deepface import DeepFace
import shutil
from collections import Counter

def draw_text_with_background_opencv(img, text, face_num, fontScale=1, thickness=2, color=(255,255,255), bg=(0,0,0)):
    padding = 10
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)
    
    # Limita o tamanho do texto ao tamanho da imagem
    max_width = img.shape[1] - 2 * padding
    if text_width > max_width:
        fontScale = fontScale * max_width / text_width
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)
    
    y = img.shape[0] - padding - face_num * (text_height + baseline + padding)  # Ajusta a posição y com base no número do rosto
    x = padding  # Início do texto ficará 10 pixels distante do canto esquerdo da imagem
    
    # Desenha um retângulo preto como plano de fundo
    cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y), bg, thickness=cv2.FILLED)
    
    # Coloca o texto na imagem
    cv2.putText(img, text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)

    return img

def select_video():
    # Abra a caixa de diálogo para selecionar o vídeo
    filepath = filedialog.askopenfilename()

    # Abra o vídeo
    cap = cv2.VideoCapture(filepath)

    # Crie um diretório para salvar os quadros extraídos
    if not os.path.exists('frames'):
        os.makedirs('frames')

    # Crie um diretório para salvar os quadros analisados
    if not os.path.exists('analyzed_frames'):
        os.makedirs('analyzed_frames')

    # Extraia os quadros do vídeo
    frame_count = 0
    emotions = []
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            # Salve o quadro como uma imagem
            cv2.imwrite(f'frames/frame{frame_count}.jpg', frame)
            frame_count += 1
        else:
            break

    # Feche o vídeo quando terminar
    cap.release()
    cv2.destroyAllWindows()

    # Analise cada quadro extraído
    for i in range(frame_count):
        # Carregue a imagem
        img_path = f'frames/frame{i}.jpg'
        img = cv2.imread(img_path)

        # Realize a análise do rosto na imagem
        results = DeepFace.analyze(img, actions = ['age', 'gender', 'race', 'emotion'], enforce_detection=False)

        # Marque a análise na imagem
        for j, result in enumerate(results):
            text = f"Face {j+1}:\nAge: {result['age']}\nGender: {result['gender']}\nRace: {result['dominant_race']}\nEmotion: {result['dominant_emotion']}"
            emotions.append(result['dominant_emotion'])
            img = draw_text_with_background_opencv(img, text, j)

        # Salve a imagem analisada
        cv2.imwrite(f'analyzed_frames/frame{i}.jpg', img)

    # Remonte o vídeo
    img_array = []
    for i in range(frame_count):
        img_path = f'analyzed_frames/frame{i}.jpg'
        img = cv2.imread(img_path)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    # Remova os diretórios de frames e analyzed_frames após o uso
    shutil.rmtree('frames')
    shutil.rmtree('analyzed_frames')

    # Analisa as emoções e exibe as porcentagens
    emotion_counter = Counter(emotions)
    total_emotions = sum(emotion_counter.values())
    emotion_percentages = {emotion: count / total_emotions * 100 for emotion, count in emotion_counter.items()}

    result_window = tk.Toplevel(root)
    result_text = tk.Text(result_window)
    result_text.pack()
    for emotion, percentage in emotion_percentages.items():
        result_text.insert(tk.END, f"Emotion: {emotion}, Percentage: {percentage:.2f}%\n")

# Crie a janela principal
root = tk.Tk()

# Crie um botão para selecionar o vídeo
button = tk.Button(root, text="Select Video", command=select_video)
button.pack()

# Inicie o loop principal
root.mainloop()
