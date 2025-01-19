import os

folder_path = 'data/game_of_thrones_heroes'
output_file = 'data/content.txt'

files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
files.sort()
biographies = []

for file in files:
    hero_name = file.replace('.txt', '')
    with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
        content = f.read()
    biographies.append(f"{hero_name}\n\n{content}")

final_content = '\n\n\n'.join(biographies)

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(final_content)