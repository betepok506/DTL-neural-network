from src.enities.training_pipeline_params import TrainingConfig, read_training_pipeline_params
from torchvision import transforms
from src.models.networks import ResNet, SiameseNet
import torch
from pathlib import Path
from PIL import Image
from tabulate import tabulate
import os
import logging
import time
import argparse
from src.utils.tensorboard_logger import get_logger
import numpy as np
from tqdm import tqdm
from src.models.model_hg import Model
import random

logger = get_logger(__name__, logging.INFO)


def get_info_by_img(filename):
    filename = filename[5:]
    t = filename.find("_crop_256x256_", 0)
    filename = filename.replace("_crop_256x256_", " ")
    ind_separator = filename.find('_', 0)
    i_j, _ = filename[t:].split('.')
    crop, layout = filename[:ind_separator], filename[ind_separator + 1:t]
    return crop, layout, i_j


def cosine_similarity(vec1, vec2):
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def pipeline_eval(**kwargs):
    config_file = kwargs['config_file']
    params = read_training_pipeline_params(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Изменение размера изображения
        transforms.ToTensor(),  # Преобразование изображения в тензор
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
    ])

    resnet = ResNet()
    model = SiameseNet(resnet)
    # model = Model('google/vit-base-patch16-224-in21k', device=device)

    if os.path.exists(params.model.path_to_model_weight):
        logger.info(f"Загрузка весов модели: {params.model.path_to_model_weight}")
        # model = torch.load(params.model.path_to_model_weight)
        logger.info("Веса модели успещно загружены!")
    else:
        logger.crutical('Файл весов не указан или не найден. Модель инициализирована случайными весами!')
        raise "Веса модели не найдены!"

    model.to(device)
    features_vectors = {}
    layout_set = set()
    times_predict = []
    crop_i_j_set = {}  # Словарь содержит индексы приток для каждого кропа
    # ПРойтись по тестовому датасету и сравнить изображения
    for folder in tqdm(os.listdir(params.dataset.path_to_test_data), desc="Итерация по каталогам", ncols=90):
        # print(folder)
        path_to_folder = os.path.join(params.dataset.path_to_test_data, folder)
        for filename in os.listdir(path_to_folder):
            path_to_file = os.path.join(path_to_folder, filename)
            crop, layout, i_j = get_info_by_img(filename)
            img = Image.open(path_to_file)
            transformed_image = transform(img)
            if not crop in crop_i_j_set:
                crop_i_j_set[crop] = {}

            if not layout in crop_i_j_set[crop]:
                crop_i_j_set[crop][layout] = set()

            start_time = time.time()
            # features = np.random.random((1, 128)).astype('float32')
            features = model.predict(transformed_image.unsqueeze(0).to(device))
            end_time = time.time()
            times_predict.append(end_time - start_time)

            layout_set.add(layout)
            crop_i_j_set[crop][layout].add(i_j)
            features_vectors[f"{crop} {layout} {i_j}"] = features.cpu().detach().numpy()
    # crop_i_j_set = {"crop3": [], "crop7": 8}
    # layout_set = {"sdfsd", "dfs", "34"}
    # results = np.zeros((len(crop_i_j_set.keys()), len(layout_set), len(layout_set), ))
    # result = [[[[] for _ in range(len(layout_set))] for _ in range(len(layout_set))] for t in
    #           range(len(crop_i_j_set.keys()))]
    positive_positive_result = []
    positive_negative_result = []
    overall_result = []
    for ind_crop, crop in enumerate(crop_i_j_set.keys()):
        result_crop = [[[] for _ in range(len(crop_i_j_set[crop]))] for _ in range(len(crop_i_j_set[crop]))]
        for ind_i_layout, layout_i in enumerate(crop_i_j_set[crop].keys()):
            for ind_j_layout, layout_j in enumerate(crop_i_j_set[crop].keys()):
                if ind_j_layout <= ind_i_layout:
                    continue

                set1 = crop_i_j_set[crop][layout_i]
                set2 = crop_i_j_set[crop][layout_j]
                for i_j in set1.intersection(set2):
                    vec1 = features_vectors[f"{crop} {layout_i} {i_j}"]
                    vec2 = features_vectors[f"{crop} {layout_j} {i_j}"]

                    dist = cosine_similarity(vec1, vec2)
                    result_crop[ind_i_layout][ind_j_layout].append(dist)
                    overall_result.append(dist)

        positive_positive_result.append(result_crop)

        # for i_j in crop_i_j_set[crop]:
        #     for ind_i_layout, layout_i in enumerate(layout_set):
        #         for ind_j_layout, layout_j in enumerate(layout_set):
        #             if ind_j_layout < ind_i_layout:
        #                 continue
        #
        #             vec1 = features_vectors[f"{crop} {layout_i} {i_j}"]
        #             vec2 = features_vectors[f"{crop} {layout_j} {i_j}"]
        #
        #             result[ind_crop][ind_i_layout][ind_j_layout].append(cosine_similarity(vec1, vec2))

    # mean_result = [[[0 for _ in range(len(layout_set))] for _ in range(len(layout_set))] for t in
    #                range(len(crop_i_j_set.keys()))]
    mean_positive_result = []
    print('-' * 20 + ' Подсчет статистики для похожих изображений ' + '-' * 20)
    for ind_crop, crop in enumerate(crop_i_j_set.keys()):
        mean_positive_result_crop = [[[] for _ in range(len(crop_i_j_set[crop]))] for _ in range(len(crop_i_j_set[crop]))]
        for ind_i_layout, layout_i in enumerate(crop_i_j_set[crop].keys()):
            for ind_j_layout, layout_j in enumerate(crop_i_j_set[crop].keys()):
                if ind_j_layout <= ind_i_layout:
                    continue

                mean_positive_result_crop[ind_i_layout][ind_j_layout] = np.mean(
                    positive_positive_result[ind_crop][ind_i_layout][ind_j_layout])

        mean_positive_result.append(mean_positive_result_crop)

    mean_negative_result = []
    print('-' * 20 + ' Подсчет статистики для НЕпохожих изображений ' + '-' * 20)
    for ind_crop, crop in tqdm(enumerate(crop_i_j_set.keys()),desc = "Итерация по кропам непохожих изображений"):
        # mean_result_crop = [[[] for _ in range(len(crop_i_j_set[crop]))] for _ in range(len(crop_i_j_set[crop]))]
        for ind_i_layout, layout_i in tqdm(enumerate(crop_i_j_set[crop].keys()), desc='Итерация по позитивным layout'):
            negative_crop = set(crop_i_j_set.keys()).difference(crop)
            for i_j in crop_i_j_set[crop][layout_i]:
                vec1 = features_vectors[f"{crop} {layout_i} {i_j}"]
                neg_crop = random.choice(list(negative_crop))
                # for neg_crop in negative_crop:
                for ind_j_layout, layout_j in enumerate(crop_i_j_set[neg_crop].keys()):
                    for i_j2 in crop_i_j_set[neg_crop][layout_j]:
                        vec2 = features_vectors[f"{neg_crop} {layout_j} {i_j2}"]
                        dist = cosine_similarity(vec1, vec2)
                        mean_negative_result.append(dist)



    # вывод статистики похожих изображений
    for ind_crop, crop in enumerate(crop_i_j_set.keys()):
        headers = [' ']
        for ind_i_layout, layout_i in enumerate(crop_i_j_set[crop].keys()):
            headers.append(layout_i)

        data = []
        for ind_i_layout, layout_i in enumerate(crop_i_j_set[crop].keys()):
            row = [layout_i]
            for ind_j_layout, layout_j in enumerate(crop_i_j_set[crop].keys()):
                row.append(mean_positive_result[ind_crop][ind_i_layout][ind_j_layout])
            data.append(row)
        print(f"Title: Crop {crop}\n{tabulate(data, headers, tablefmt='grid')}")
        print('\n')

    print(f'Средняя дистанция для похожих изображений: {np.mean(overall_result)}')
    print(f'Средняя дистанция для НЕпохожих изображений: {np.mean(mean_negative_result)}')
    print(f'Среднее время предсказания модели: {np.mean(times_predict)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_file", help="")
    parser.add_argument("--output_dir", help="")
    args = parser.parse_args()
    pipeline_eval(**vars(args))
