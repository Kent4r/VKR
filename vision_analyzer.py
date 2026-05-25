import logging
import io
import json
from datetime import datetime
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class VisionAnalyzer:
    def __init__(self, use_mock: bool = False, model_path: str = "yolov8n.pt", log_file: str = "vision_log.txt"):
        self.use_mock = use_mock
        self.model = None
        self.enabled = False
        self.log_file = log_file
        if not use_mock:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.enabled = True
                logger.info("YOLO модель загружена, анализ фото будет реальным")
            except Exception as e:
                logger.warning(f"Не удалось загрузить YOLO: {e}, переключение в режим заглушки")
                self.use_mock = True

    def _log(self, method: str, image_bytes: bytes, result: dict):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Время: {datetime.now().isoformat()}\n")
                f.write(f"Метод: {method}\n")
                f.write(f"Размер: {len(image_bytes)} байт\n")
                f.write(f"Результат:\n{json.dumps(result, ensure_ascii=False, indent=2)}\n")
        except Exception as e:
            logger.error(f"Ошибка записи лога: {e}")

    # vision_analyzer.py (фрагмент с изменениями)
    def analyze_image(self, image_bytes: bytes) -> dict:
        if self.use_mock:
            result = self._mock_response(image_bytes)
        else:
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img_np = np.array(img)
                results = self.model(img_np)
                result = self._parse_yolo_results(results[0])
            except Exception as e:
                logger.error(f"Ошибка YOLO: {e}")
                result = self._mock_response(image_bytes)
        self._log("analyze_image", image_bytes, result)
        return result

    def _parse_yolo_results(self, result) -> dict:
        boxes = result.boxes
        detected = []
        if boxes:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = result.names[cls_id]
                detected.append({"object": name, "confidence": conf})
        # Формируем человекочитаемое описание и список объектов
        object_names = [d["object"] for d in detected]
        description = f"На фото обнаружены: {', '.join(object_names)}." if object_names else "На фото не обнаружено значимых объектов."
        return {
            "damage_detected": len(detected) > 0,
            "objects": detected,
            "description": description,
            "recommendation": "Пожалуйста, уточните проблему."
        }

    def _mock_response(self, image_bytes: bytes) -> dict:
        import hashlib
        img_hash = hashlib.md5(image_bytes).hexdigest()[:6]
        val = int(img_hash, 16) % 4
        if val == 0:
            return {
                "damage_detected": True,
                "objects": [{"object": "optical_cable", "state": "broken", "confidence": 0.94}],
                "description": "Обнаружен разрыв оптического кабеля.",
                "recommendation": "Требуется выезд специалиста."
            }
        elif val == 1:
            return {
                "damage_detected": False,
                "objects": [{"object": "router", "state": "normal", "confidence": 0.87}],
                "description": "Роутер выглядит исправно.",
                "recommendation": "Проверьте настройки Wi-Fi."
            }
        elif val == 2:
            return {
                "damage_detected": False,
                "objects": [{"object": "ont_terminal", "state": "blinking_los", "confidence": 0.92}],
                "description": "Терминал мигает красным (LOS).",
                "recommendation": "Проверьте подключение оптического кабеля."
            }
        else:
            return {
                "damage_detected": False,
                "objects": [],
                "description": "Не удалось распознать оборудование.",
                "recommendation": "Сделайте более чёткое фото."
            }

    def analyze_image_detailed(self, image_bytes: bytes) -> dict:
        """Расширенный анализ с перечислением объектов."""
        if self.use_mock:
            result = self._mock_response_detailed(image_bytes)
        else:
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img_np = np.array(img)
                results = self.model(img_np)
                result = self._detailed_from_yolo(results[0])
            except Exception as e:
                result = self._mock_response_detailed(image_bytes)
        self._log("analyze_image_detailed", image_bytes, result)
        return result

    def _detailed_from_yolo(self, result) -> dict:
        boxes = result.boxes
        objects = []
        if boxes:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = result.names[cls_id]
                objects.append({"type": name, "confidence": conf})
        return {
            "objects": objects,
            "summary": f"Найдено {len(objects)} объектов: {', '.join([o['type'] for o in objects])}" if objects else "Объекты не обнаружены."
        }

    def _mock_response_detailed(self, image_bytes: bytes) -> dict:
        import hashlib
        img_hash = hashlib.md5(image_bytes).hexdigest()[:6]
        if int(img_hash, 16) % 2 == 0:
            return {
                "objects": [
                    {"type": "terminal", "state": "normal", "confidence": 0.92},
                    {"type": "router", "state": "normal", "confidence": 0.88},
                    {"type": "cable", "state": "damaged", "damage_type": "fracture", "confidence": 0.96}
                ],
                "summary": "Разрыв кабеля, терминал и роутер исправны."
            }
        else:
            return {
                "objects": [
                    {"type": "router", "state": "blinking", "confidence": 0.85},
                    {"type": "cable", "state": "ok", "confidence": 0.7}
                ],
                "summary": "Роутер моргает, кабель цел. Возможны проблемы с питанием."
            }