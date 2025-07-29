from pathlib import Path
from treelib import Tree

def add_to_tree(tree, path, parent=None):
    """Добавляет элементы в дерево, игнорируя содержимое .git, venv и других скрытых папок."""
    name = path.name
    node_id = str(path)
    
    # Создаем узел для текущего пути
    tree.create_node(tag=name, identifier=node_id, parent=parent)
    
    # Если это директория И НЕ скрытая/venv — рекурсивно добавляем содержимое
    if path.is_dir():
        # Правило для исключения: папка начинается с точки (.git) или это venv
        if not (name.startswith('.') or name.startswith('__') or name == 'venv'):
            for item in path.iterdir():
                add_to_tree(tree, item, parent=node_id)

def main():
    # Инициализация дерева
    tree = Tree()
    root_path = Path("C:\\Users\\novos\\Desktop\\GitHub\\grokking-machine-learning")
    
    # Строим дерево
    add_to_tree(tree, root_path)
    
    # Выводим результат
    tree.show()

if __name__ == "__main__":
    main()
