import requests
from bs4 import BeautifulSoup

episodes = [
    'Зима_близко', 'Королевский_тракт', 'Лорд_Сноу', 'Калеки,_бастарды_и_сломанные_вещи', 'Волк_и_лев', 'Золотая_корона', 'Победа_или_смерть', 'Острый_конец', 'Бейлор', 'Пламя_и_кровь',
    'Север_помнит', 'Ночные_земли', 'Что_мертво,_умереть_не_может', 'Костяной_сад', 'Призрак_Харренхола', 'Старые_Боги_и_Новые', 'Человек_без_чести', 'Принц_Винтерфелла', 'Черноводная', 'Валар_Моргулис',
    'Валар_Дохэйрис', 'Тёмные_крылья,_тёмные_слова', 'Аллея_Наказания', 'И_теперь_его_дозор_окончен', 'Поцелованная_огнём', 'Восхождение', 'Медведь_и_прекрасная_дева', 'Младшие_сыновья', 'Дожди_в_Кастамере', 'Миса',
    'Два_меча', 'Лев_и_роза', 'Разрушительница_цепей', 'Верный_клятве', 'Первый_своего_имени', 'Законы_богов_и_людей', 'Пересмешник', 'Гора_и_Змей', 'Дозорные_на_Стене', 'Дети',
    'Грядущие_войны', 'Чёрно-белый_дом', 'Его_Воробейшество', 'Сыны_Гарпии', 'Убей_мальчишку', 'Непреклонные,_несгибаемые,_несдающиеся', 'Дар', 'Суровый_Дом', 'Танец_драконов', 'Милосердие_матери',
    'Красная_женщина', 'Дом', 'Клятвопреступник', 'Книга_Неведомого', 'Дверь', 'Кровь_моей_крови', 'Сломленный', 'Никто_(серия)', 'Битва_бастардов', 'Ветра_зимы',
    'Драконий_Камень_(серия)', 'Бурерожденная', 'Правосудие_королевы', 'Трофеи_войны', 'Восточный_дозор', 'За_Стеной', 'Дракон_и_волк',
    'Винтерфелл_(серия)', 'Рыцарь_Семи_Королевств', 'Долгая_ночь_(серия)', 'Последние_из_Старков', 'Колокола', 'Железный_трон_(серия)'     
]


def fetch_episode_content(url, num):
    # Настройка заголовков для запроса
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Получаем страницу
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.find('h1', class_='page-header__title').text.strip()
        content = soup.find('div', class_='mw-parser-output')
        
        # Флаги для определения нужной секции
        start_found = False
        plot_text = []
        
        for element in content.children:
            if element.name == 'h2' and 'Краткое содержание' in element.text:
                start_found = True
                continue
                
            if element.name == 'h2' and 'Галерея' in element.text:
                break
                
            if start_found and element.name == 'p':
                text = element.get_text().strip()
                if text:
                    plot_text.append(text)
        
        # Объединяем весь текст и сохраняем
        full_text = '\n\n'.join(plot_text)
        
        filename = f"game_of_thrones_episodes/{num}.{title}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_text)
            
        print(f"Содержание серии сохранено в файл: {filename}")
        
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

num = 1
for episode in episodes:
    url = "https://gameofthrones.fandom.com/ru/wiki/" + episode
    fetch_episode_content(url, num)
    num += 1