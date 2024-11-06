### Librairies à installer
# pip install deepface streamlit pillow matplotlib altair opencv-python
# pip install watchdog tensorflow
# pip install tf-keras
# pip install facenet-pytorch
# pip install mediapipe
# pip install facenet-pytorch
# pip install ultralytics
# pip install opencv-contrib-python
# pip install opencv-python
# pip install yt-dlp

import streamlit as st
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
from fer import FER

# Configuration de la page Streamlit
st.set_page_config(page_title="Comparatif entre DeepFace et FER+")

# Constantes
BACKENDS = ['opencv', 'ssd', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet', 'centerface']
MODELS = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepID", "ArcFace", "SFace", "GhostFaceNet"]

def draw_face(img_orig, face, detection_time):
    image = img_orig.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    if isinstance(face, dict) and all(k in face for k in ['x', 'y', 'w', 'h']):
        x, y, w, h = face['x'], face['y'], face['w'], face['h']
    elif 'facial_area' in face:
        x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
    else:
        st.error("Format de coordonnées de visage inattendu dans `face`: " + str(face))
        return image

    draw.rectangle([(x, y), (x + w, y + h)], outline="green", width=2)
    draw.text((10, 10), f"Temps de détection: {detection_time:.2f}s", fill="red", font=font)
    return image

def draw_emotions_and_face(img_orig, face, emotions, detection_time):
    image = img_orig.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    if isinstance(face, dict) and all(k in face for k in ['x', 'y', 'w', 'h']):
        x, y, w, h = face['x'], face['y'], face['w'], face['h']
    elif 'facial_area' in face:
        x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
    else:
        st.error("Format de coordonnées de visage inattendu dans `face`: " + str(face))
        return image

    draw.rectangle([(x, y), (x + w, y + h)], outline="green", width=2)
    for idx, (emotion, score) in enumerate(emotions.items()):
        draw.text((x, y + 10 + idx * 18), f"{emotion}: {score:.2f}", fill="blue", font=font)
    draw.text((10, 10), f"Temps de détection: {detection_time:.2f}s", fill="red", font=font)
    return image

def detect_faces_only(image, model_name):
    try:
        start_time = time.monotonic()
        embedding = DeepFace.represent(img_path=image,
                                       model_name=model_name,
                                       enforce_detection=False)
        detection_time = time.monotonic() - start_time
        return embedding, detection_time
    except Exception as e:
        st.error(f"Erreur lors de la détection avec le modèle {model_name}: {str(e)}")
        return None, 0


### fonction pour isoler l'analyse avec le modèle VGG-face + Beckend (Retinaface ou Yolov8) + émotions
### fonction Deep-face - modèle VGG-face - Backend retinaface + Détection d'émotions
def detect_faces_and_emotions_retinaface(image):
    try:
        start_time = time.monotonic()  # Début du chronométrage
        resultats = DeepFace.analyze(img_path=image,
                                     actions=['emotion'],
                                     detector_backend="retinaface",
                                     enforce_detection=False,
                                     align=False,
                                     silent=True)
        detection_time = time.monotonic() - start_time  # Fin du chronométrage et calcul du temps

        if isinstance(resultats, list):
            resultats = resultats[0]

        return resultats, detection_time  # Retourne également le temps de détection

    except Exception as e:
        st.error(f"Erreur lors de la détection avec RetinaFace: {str(e)}")
        return None, 0

### fonction Deep-face - modèle VGG-face - Backend Yolov8 + Détection d'émotions
def detect_faces_and_emotions_yolov8(image):
    try:
        start_time = time.monotonic()  # Début du chronométrage
        resultats = DeepFace.analyze(img_path=image,
                                     actions=['emotion'],
                                     detector_backend="yolov8",
                                     enforce_detection=False,
                                     align=False,  # option d'alignement facial
                                     silent=True)
        detection_time = time.monotonic() - start_time  # Fin du chronométrage et calcul du temps

        if isinstance(resultats, list):
            resultats = resultats[0]

        return resultats, detection_time  # Retourne également le temps de détection

    except Exception as e:
        st.error(f"Erreur lors de la détection avec YOLOv8: {str(e)}")
        return None, 0

### fonction Deep-face - modèle VGG-face - Backend MTCNN + Détection d'émotions
def detect_faces_and_emotions_mtcnn(image):
    try:
        start_time = time.monotonic()  # Début du chronométrage
        resultats = DeepFace.analyze(img_path=image,
                                     actions=['emotion'],  # Spécifie l'analyse émotionnelle uniquement
                                     detector_backend="mtcnn",  # Utilise MTCNN comme backend
                                     enforce_detection=False,
                                     align=False,  # option d'alignement facial
                                     silent=True)
        detection_time = time.monotonic() - start_time  # Fin du chronométrage et calcul du temps

        if isinstance(resultats, list):
            resultats = resultats[0]

        return resultats, detection_time  # Retourne également le temps de détection

    except Exception as e:
        st.error(f"Erreur lors de la détection avec MTCNN: {str(e)}")
        return None, 0


def create_result_grid(images, titles, n_cols=3):
    n_rows = (len(images) + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(15, 5 * n_rows))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.3)

    for ax, im, title in zip(grid, images, titles):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis('off')

    for ax in grid[len(images):]:
        ax.axis('off')

    return fig

def main():
    st.title("Analyse de visages avec DeepFace et FER+")

    uploaded_image = st.file_uploader("Choisissez une image", type=["jpg", "png"])

    if uploaded_image:
        img = Image.open(uploaded_image)
        img_array = np.array(img)

        st.markdown("""
        ## Test 1 : Comparatif des modèles pour la détection de visages (sans analyse des émotions)
        Nous allons d'abord tester les différents modèles disponibles dans DeepFace pour la détection de visages, sans effectuer d'analyse d'émotions.
        """)
        test_models(img, img_array)

        st.markdown("""
        ### Test 2 : Comparatif des différents backends pour la détection des visages avec VGG-Face
        Nous allons maintenant tester plusieurs backends disponibles dans DeepFace pour voir comment ils affectent la détection des visages. 
        Le modèle utilisé pour l'analyse des visages reste **VGG-Face**.
        """)
        test_backends(img, img_array)

        st.markdown("""
        ### Test 3 : Comparatif des différents backends pour la détection des visages avec Facenet512
        Cette étape vise à tester le modèle **Facenet512** avec chaque backend pour évaluer les performances de détection de visages.
        """)
        test_backends_facenet512(img, img_array)

        st.markdown("""
        ### Test 4 : Analyse des émotions avec VGG-Face et backend RetinaFace
        L'analyse des émotions est réalisée avec le modèle **VGG-Face** et le backend **RetinaFace**.
        """)
        resultats, detection_time = detect_faces_and_emotions_retinaface(img_array)

        if resultats:
            emotions = resultats.get('emotion', {})
            face = resultats['region']
            img_with_emotions = draw_emotions_and_face(img, face, emotions, detection_time)
            st.image(img_with_emotions, caption="Résultat de l'analyse des émotions avec VGG-Face et RetinaFace")
            df_results = pd.DataFrame([emotions])
            df_results['backend'] = 'retinaface'
            df_results['detection_time'] = detection_time
            st.dataframe(df_results)
        else:
            st.warning("Aucun visage détecté avec RetinaFace.")

        st.markdown("""
        ### Test 5 : Analyse des émotions avec VGG-Face et YOLOv8
        L'analyse des émotions est réalisée avec le modèle **VGG-Face** et le backend **YOLOv8**.
        """)
        resultats, detection_time = detect_faces_and_emotions_yolov8(img_array)

        if resultats:
            emotions = resultats.get('emotion', {})
            face = resultats['region']
            img_with_emotions = draw_emotions_and_face(img, face, emotions, detection_time)
            st.image(img_with_emotions, caption="Résultat de l'analyse des émotions avec VGG-Face et YOLOv8")
            df_results = pd.DataFrame([emotions])
            df_results['backend'] = 'yolov8'
            df_results['detection_time'] = detection_time
            st.dataframe(df_results)
        else:
            st.warning("Aucun visage détecté avec YOLOv8.")

        # Test 6 : Analyse des émotions avec VGG-Face et MTCNN
        st.markdown("""
        ### Test 6 : Analyse des émotions avec VGG-Face et MTCNN
        L'analyse des émotions est réalisée avec le modèle **VGG-Face** et le backend **MTCNN**.
        """)
        resultats, detection_time = detect_faces_and_emotions_mtcnn(img_array)

        # Affichage des résultats du test 6
        if resultats:
            emotions = resultats.get('emotion', {})
            face = resultats['region']
            img_with_emotions = draw_emotions_and_face(img, face, emotions, detection_time)
            st.image(img_with_emotions, caption="Résultat de l'analyse des émotions avec VGG-Face et MTCNN")
            df_results = pd.DataFrame([emotions])
            df_results['backend'] = 'mtcnn'
            df_results['detection_time'] = detection_time
            st.dataframe(df_results)
        else:
            st.warning("Aucun visage détecté avec MTCNN.")

        st.markdown("""
        ### Test 7 : Analyse des émotions avec FER
        Nous allons utiliser FER+ pour l'analyse des émotions. FER utilise le backend MTCNN.
        """)

        # Appel de la fonction pour obtenir les résultats et le temps de détection
        fer_result, detection_time_fer = analyse_with_ferplus(img_array)

        # Affichage des résultats
        if fer_result:
            for face in fer_result:
                bbox = face['box']
                emotions_fer = face['emotions']
                img_with_emotions_fer = draw_emotions_and_face(
                    img,
                    {'x': bbox[0], 'y': bbox[1], 'w': bbox[2], 'h': bbox[3]},
                    emotions_fer,
                    detection_time_fer
                )
                st.image(img_with_emotions_fer, caption="Résultat de l'analyse avec FER+")

                # Création et affichage du DataFrame des résultats
                df_results_fer = pd.DataFrame([emotions_fer])
                df_results_fer['model'] = 'FER+'
                df_results_fer['backend'] = 'mtcnn'
                df_results_fer['detection_time'] = detection_time_fer
                st.dataframe(df_results_fer)
        else:
            st.warning("Aucun visage détecté avec FER+.")
#####

def test_models(img, img_array):
    images_models = []
    titles_models = []
    all_results = []

    for model in MODELS:
        faces, detection_time = detect_faces_only(img_array, model)

        if faces:
            img_with_face = draw_face(img, faces[0], detection_time)
            images_models.append(img_with_face)
            titles_models.append(f"{model}, T={detection_time:.2f}s")
            all_results.append({"model": model, "detection_time": detection_time})
        else:
            st.write(f"Aucun visage détecté avec le modèle {model}")

    if images_models:
        st.pyplot(create_result_grid(images_models, titles_models))
        df_model_results = pd.DataFrame(all_results)
        st.dataframe(df_model_results)
    else:
        st.warning("Aucun modèle n'a réussi à détecter un visage.")

def test_backends(img, img_array):
    images_backend = []
    titles_backend = []
    all_results = []

    for backend in BACKENDS:
        start_time = time.monotonic()
        try:
            resultats = DeepFace.analyze(img_path=img_array,
                                         detector_backend=backend,
                                         enforce_detection=False,
                                         align=False,  # option d'alignement facial
                                         silent=True)

            detection_time = time.monotonic() - start_time

            if isinstance(resultats, list):
                resultats = resultats[0]
            if 'region' in resultats:
                face = resultats['region']
                img_with_face = draw_face(img, face, detection_time)
                images_backend.append(img_with_face)
                titles_backend.append(f"{backend.capitalize()}, T={detection_time:.2f}s")
                all_results.append({"backend": backend, "detection_time": detection_time})
        except Exception as e:
            st.write(f"Erreur avec le backend {backend}: {str(e)}")

    if images_backend:
        st.pyplot(create_result_grid(images_backend, titles_backend))
        df_backend_results = pd.DataFrame(all_results)
        st.dataframe(df_backend_results)
    else:
        st.warning("Aucun backend n'a réussi à détecter un visage.")


def test_backends_facenet512(img, img_array):
    images_backend = []
    titles_backend = []
    all_results = []

    for backend in BACKENDS:
        start_time = time.monotonic()
        try:
            # Utiliser `DeepFace.represent` pour obtenir les embeddings avec Facenet512
            faces = DeepFace.represent(img_path=img_array,
                                       model_name="Facenet512",
                                       detector_backend=backend,
                                       align=False,  # option d'alignement facial
                                       enforce_detection=False)
            detection_time = time.monotonic() - start_time

            # Vérifier si un visage est détecté (présence d'embeddings)
            if faces:
                img_with_face = draw_face(img, faces[0], detection_time)
                images_backend.append(img_with_face)
                titles_backend.append(f"{backend.capitalize()}, T={detection_time:.2f}s")
                all_results.append({"backend": backend, "detection_time": detection_time})
            else:
                st.write(f"Aucun visage détecté avec le backend {backend}")

        except Exception as e:
            st.write(f"Erreur avec le backend {backend}: {str(e)}")

    if images_backend:
        st.pyplot(create_result_grid(images_backend, titles_backend))
        df_backend_results = pd.DataFrame(all_results)
        st.dataframe(df_backend_results)
    else:
        st.warning("Aucun backend n'a réussi à détecter un visage.")


def analyse_with_ferplus(img_array):
    detector = FER(mtcnn=True)  # Utilisation de FER avec le backend MTCNN
    start_time = time.monotonic()
    fer_result = detector.detect_emotions(img_array)
    detection_time_fer = time.monotonic() - start_time

    return fer_result, detection_time_fer  # Retourne les résultats et le temps de détection


if __name__ == "__main__":
    main()
