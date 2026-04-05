import os
import re

def create_from_tree(tree_text, base_dir="."):
    lines = tree_text.strip().split('\n')
    current_path = []
    
    for line in lines:
        # Определяем уровень вложенности
        indent = len(re.match(r'^[│├└\─\s]*', line).group())
        line = line.strip('│├└─ ')
        
        if line.endswith('/'):  # Папка
            folder_name = line.rstrip('/')
            path_stack = current_path[:indent//2]
            full_path = os.path.join(base_dir, *path_stack, folder_name)
            os.makedirs(full_path, exist_ok=True)
            current_path = current_path[:indent//2] + [folder_name]
        else:  # Файл
            path_stack = current_path[:indent//2]
            dir_path = os.path.join(base_dir, *path_stack)
            os.makedirs(dir_path, exist_ok=True)
            full_path = os.path.join(dir_path, line)
            with open(full_path, 'w') as f:
                pass  # Создаем пустой файл

# Пример использования
tree_structure = """
├── .env
├── main.py
├── orchestrator.py
├── gigachat_client.py
├── prompts.py
"""

create_from_tree(tree_structure)