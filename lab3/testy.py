
import random
import pandas as pd


def get_user_inputs():
    user_data = []
    for _ in range(3):  # Проводим цикл для 10 человек
        # Запрашиваем пол пользователя
        gender = input("Введите пол (мужчина/женщина): ").strip().lower()
        while gender not in ['мужчина', 'женщина']:
            gender = input("Некорректный ввод. Пожалуйста, введите 'мужчина' или 'женщина': ").strip().lower()

        # Запрашиваем дату рождения у пользователя
        birth_date = input("Введите дату рождения (в формате ГГГГ-ММ-ДД): ").strip()

        user_data.append((gender, birth_date))

    return user_data


def generate_random_values():
    # Генерируем случайные данные
    R = random.uniform(150, 200)  # Рост в см
    OT = random.uniform(60, 120)  # Окружность талии в см
    OB = random.uniform(80, 140)  # Окружность бедер в см
    V = random.uniform(50, 120)  # Вес в кг
    return R, OT, OB, V


def calculate_biological_age(gender, R, OT, OB, V, RL):
    if gender == 'женщина':
        RL -= 18
        kss = ((OT * V) / (OB * (R ** 2) * (14.7 + 0.26 * RL + 0.01 * RL)))
        bio_age = kss * RL + 18
    else:
        RL -= 21
        kss = ((OT * V) / (OB * (R ** 2) * (17.2 + 0.31 * RL + 0.0012 * RL)))
        bio_age = kss * RL + 21
    return bio_age


def main():
    # Получаем пользовательские вводы
    user_data = get_user_inputs()

    # Список для хранения данных о результатах
    results = []

    for gender, birth_date in user_data:
        # Устанавливаем значение РЛ в зависимости от пола
        RL = 18 if gender == 'женщина' else 21

        # Генерируем случайные значения для роста, окружности талии, окружности бедер и веса
        R, OT, OB, V = generate_random_values()
#
        # Рассчитываем биологический возраст
        bio_age = calculate_biological_age(gender, R, OT, OB, V, RL)

        # Добавляем данные в результаты
        results.append({
            'Пол': gender,
            'Дата рождения': birth_date,
            'Рост (м)': R,
            'Окружность талии (см)': OT,
            'Окружность бедер (см)': OB,
            'Вес (кг)': V,
            'Биологический возраст': bio_age
        })

#     # Создаем DataFrame для записи в Excel
    df = pd.DataFrame(results)
    print(df)
    # Записываем данные в Excel
    #output_file = "biological_age_calculation.xlsx"
    #df.to_excel(output_file, index=False)



if __name__ == "__main__":
    main()
