# Flower Similarity Search

Модель для поиска 5 наиболее схожих изображений цветов из датасета по заданному изображению.

## Описание проекта

Проект реализует систему поиска похожих изображений на основе:
- Извлечения признаков с помощью предобученной DINOv2
- Сравнения векторных представлений изображений
- Поиска 5 ближайших соседей по косинусному расстоянию

## Датасет

В проекте используется датасет из 5 видов цветов:
1. daisy
2. dandelion
3. rose
4. sunflower
5. tulip


## Требования

Для работы проекта необходимо:
- Python 3.11+
- Установленные зависимости из `requirements.txt`: