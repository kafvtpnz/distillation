
Проект демонстрирует возможность дистилляции

В качестве донора используется трансформер Bert обученый на решение задачи классификации комментариев с сайтов Двач и Пикабу
датасет Russian Language Toxic Comments Dataset https://www.kaggle.com/blackmoon/russian-language-toxic-comments

Перед началом работы нужно скачать (https://drive.google.com/file/d/1e4ZLyUGU22RwjSDHnTKHBe0XwsPr6xl1/view?usp=sharing)
и положить файл весов Bert-а в каталог model_bert



Обучение CNN по обученному Bert 

distillation.py

Параметры:
--alpha, type=float, default=0.5  коэффициент, задающий соотношение loss на данных к loss на Bert-е при обучении CNN
--batch_size, type=int, default=4 размер батча
--epochs, type=int, default=10,   эпох обучения
--lr, type=float, default=0.002,  стартовая скорость обучения
--max_len, type=int, default=50,  длина последовательности, которая энкодится в CNN


Предсказание Bert
bert_pred.py --text "ВАШ ТЕКСТ"

Предсказание CNN (обученная сеть в каталоге model_cnn после процедуры дисцилляции)
cnn_pred.py --text "ВАШ ТЕКСТ"


В пределах 10 эпох обучения лучший результат на валидации
AccuracyCNN: 0.739612188365651, AccuracyBert: 0.8961218836565097
