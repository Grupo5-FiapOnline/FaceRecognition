import face_recognition
import cv2
import os
import numpy as np

dataset_dir = "dataset"

known_encodings = []
known_names = []

# Treinamento com 4 fotos de cada pessoa
for person_name in os.listdir(dataset_dir):
    if person_name.startswith('.'):
        # Ignorar arquivos ocultos/de sistema
        continue
    person_folder = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_folder):
        continue
    print(person_folder)
    images = os.listdir(person_folder)
    print(images)
    train_images = images[:4]  # usa 4 para treino
    for img_name in train_images:
        img_path = os.path.join(person_folder, img_name)
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person_name)

# Teste na 5ª foto
print("Testando reconhecimento...")
for person_name in os.listdir(dataset_dir):
    test_img_path = os.path.join(dataset_dir, person_name, "5.jpg")
    test_img = face_recognition.load_image_file(test_img_path)
    test_encodings = face_recognition.face_encodings(test_img)

    if len(test_encodings) == 0:
        print(f"Nenhum rosto detectado em {test_img_path}")
        continue

    test_encoding = test_encodings[0]
    results = face_recognition.compare_faces(known_encodings, test_encoding)
    face_distances = face_recognition.face_distance(
        known_encodings, test_encoding)

    best_match_index = np.argmin(face_distances)
    if results[best_match_index]:
        print(
            f"Reconhecido: {known_names[best_match_index]} como {person_name}")
    else:
        print(f"Falha ao reconhecer {person_name}")
