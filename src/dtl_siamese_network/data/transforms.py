from torchvision import transforms
import random


class RandomRotationDegree:
    def __call__(self, img):
        angle = random.uniform(0, 360)  # Случайный угол от 0 до 360 градусов
        return transforms.functional.rotate(img, angle)


def get_train_transforms(mean=0.1307, std=0.3081):
    transform_train = transforms.Compose([
        RandomRotationDegree(),  # Случайный поворот на любой угол
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # Случайные изменения яркости, контрастности, насыщенности и оттенка
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Случайное изменение размера и обрезка
        transforms.RandomHorizontalFlip(),  # Случайное горизонтальное отражение
        transforms.RandomVerticalFlip(),  # Случайное вертикальное отражение
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    return transform_train


def get_test_transforms(mean=0.1307, std=0.3081):
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    return transform_test
