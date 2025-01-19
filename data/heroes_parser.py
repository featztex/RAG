import requests
from bs4 import BeautifulSoup
import os

names = [
    "Эддард_Старк", "Кейтилин_Старк", "Робб_Старк", "Санса_Старк", "Арья_Старк", "Бран_Старк", "Джон_Сноу", "Теон_Грейджой", "Бринден_Талли", "Эдмур_Талли",
    "Тайвин_Ланнистер", "Серсея_Ланнистер", "Джейме_Ланнистер", "Тирион_Ланнистер", "Джоффри_Баратеон", "Томмен_Баратеон", "Бронн", "Шая", "Григор_Клиган", "Сандор_Клиган",
    "Дейенерис_Таргариен", "Визерис_Таргариен", "Эйрис_II_Таргариен", "Рейгар_Таргариен", "Дрого", "Джорах_Мормонт", "Серый_Червь", "Миссандея", "Барристан_Селми", "Драконы",
    "Роберт_Баратеон", "Джендри", "Станнис_Баратеон", "Мелисандра", "Давос_Сиворт", "Ренли_Баратеон", "Маргери_Тирелл", "Лорас_Тирелл", "Оленна_Тирелл", "Бриенна_Тарт",
    "Рамси_Болтон", "Русе_Болтон", "Уолдер_Фрей", "Король_Ночи", "Эурон_Грейджой", "Петир_Бейлиш", "Варис", "Сэмвелл_Тарли", "Оберин_Мартелл", "Ходор"
]

def parse_biography(url):

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Получаем страницу
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Получаем имя персонажа для названия файла
        character_name = soup.find('h1', class_='page-header__title').text.strip()
        
        # Находим секцию с биографией
        content = soup.find('div', class_='mw-parser-output')
        start_found = False
        biography_text = []
        
        for element in content.children:

            if element.name == 'h2' and 'Биография' in element.text:
                start_found = True
                continue

            if element.name == 'h2' and 'Интересные факты' in element.text:
                break
                
            # Если мы в нужной секции, собираем текст
            if start_found and element.name == 'p':
                text = element.get_text().strip()
                if text:
                    biography_text.append(text)
        
        # Объединяем весь текст
        full_text = '\n\n'.join(biography_text)
        
        # Сохраняем в файл
        output_folder = 'data/game_of_thrones_heroes'
        os.makedirs(output_folder, exist_ok=True)
        filename = os.path.join(output_folder, f"{character_name}.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_text)
            
        print(f"Биография сохранена в файл: {filename}")
        
        
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

base_url = "https://gameofthrones.fandom.com/ru/wiki/"
for name in names:
    url = base_url + name
    parse_biography(url)