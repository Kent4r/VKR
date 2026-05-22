# vision_analyzer.py
import logging
import io
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class VisionAnalyzer:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.enabled = False
        self.model = None
        try:
            from ultralytics import YOLO
            # Загружаем модель
            self.model = YOLO(model_path)
            self.enabled = True
            logger.info(f"Модель YOLO загружена из {model_path}")
        except ImportError:
            logger.warning("Библиотека ultralytics не установлена. Анализ фото работать не будет.")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")

    def analyze_image(self, image_bytes: bytes) -> dict:

        # Анализирует изображение и возвращает результат.
        # Для демонстрации: если модель обнаруживает объекты с уверенностью >0.5,
        # считаем, что есть потенциальное повреждение.

        if not self.enabled:
            return {"damage_detected": False, "message": "Модуль CV отключён (не установлен ultralytics)"}

        try:
            # Открываем изображение
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Преобразуем PIL в numpy
            img_np = np.array(img)

            # Инференс
            results = self.model(img_np)
            result = results[0]

            # Извлекаем детекции
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return {
                    "damage_detected": False,
                    "message": "На фото не обнаружено объектов, похожих на повреждение",
                    "confidence": 0.0
                }

            # Для демонстрации берём самую уверенную детекцию
            confidences = boxes.conf.cpu().numpy()
            max_conf = float(confidences.max()) if len(confidences) > 0 else 0.0
            class_ids = boxes.cls.cpu().numpy()
            # Имена классов для COCO (можно вывести для отладки)
            # names = self.model.names

            # Логика: считаем, что любой обнаруженный объект с высокой вероятностью
            # может быть признаком проблемы (заглушка). В реальности нужно,
            # чтобы модель была обучена на классах "fracture", "bend" и т.д.
            if max_conf > 0.5:
                return {
                    "damage_detected": True,
                    "damage_type": "unknown",   # здесь мог бы быть класс из модели
                    "confidence": float(max_conf),
                    "description": f"Обнаружен объект с уверенностью {max_conf:.2f}. Возможно, это повреждение оптического волокна. Требуется проверка оператором."
                }
            else:
                return {
                    "damage_detected": False,
                    "message": f"Обнаружены объекты, но уверенность太低 ({max_conf:.2f})",
                    "confidence": max_conf
                }

        except Exception as e:
            logger.error(f"Ошибка анализа фото: {e}")
            return {"damage_detected": False, "error": str(e)}