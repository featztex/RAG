import os

folder_path = 'data/game_of_thrones_episodes'
output_file = 'content.txt'

files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# Сортируем файлы по номеру серии
files.sort(key=lambda x: int(x.split('.')[0]))

# Объединяем содержимое файлов
with open(output_file, 'w', encoding='utf-8') as outfile:
    for file in files:

        series_number = file.split('.')[0]
        
        with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as infile:
            content = infile.read()
        
        outfile.write(f'Серия {series_number}\n')
        outfile.write(content + '\n' + '\n')
