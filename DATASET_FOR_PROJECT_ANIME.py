import kagglehub
import os
import shutil

# Определяем основную папку для датасетов
datasets_dir = '../datasets'
os.makedirs(datasets_dir, exist_ok=True)

# Скачиваем Anime Faces dataset
print("Скачивание Anime Faces dataset...")
anime_faces_path = kagglehub.dataset_download("splcher/animefacedataset")
print("Путь к файлам Anime Faces dataset:", anime_faces_path)

# Скачиваем CelebA dataset
print("\nСкачивание CelebA dataset...")
celeba_path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
print("Путь к файлам CelebA dataset:", celeba_path)

print("\nПеремещение файлов в директорию 'datasets'...")

# Перемещаем файлы из Anime Faces в datasets
if anime_faces_path:
    for item in os.listdir(anime_faces_path):
        source_path = os.path.join(anime_faces_path, item)
        destination_path = os.path.join(datasets_dir, item)
        try:
            if os.path.isdir(source_path):
                shutil.move(source_path, destination_path)
            else:
                shutil.move(source_path, destination_path)
        except Exception as e:
            print(f"Ошибка при перемещении {item} из Anime Faces: {e}")

# Перемещаем файлы из CelebA в datasets
if celeba_path:
    for item in os.listdir(celeba_path):
        source_path = os.path.join(celeba_path, item)
        destination_path = os.path.join(datasets_dir, item)
        try:
            if os.path.isdir(source_path):
                shutil.move(source_path, destination_path)
            else:
                shutil.move(source_path, destination_path)
        except Exception as e:
            print(f"Ошибка при перемещении {item} из CelebA: {e}")

print("\nПеремещение файлов завершено.")