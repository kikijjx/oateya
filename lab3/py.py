def calculate_biological_age(height: float, waist_size: float, hip_size: float, weight: float, steps: int, sex: bool) -> float:
    height = height / 100
    if sex:
        RL = 21
        KSS = (waist_size * weight) / (hip_size * height**2 * (17.2 + 0.31 * RL + 0.0012 + RL))
        biological_age = KSS * RL + 21
    else:
        RL = 18
        KSS = (waist_size * weight) / (hip_size * height**2 * (14.7 + 0.26 * RL + 0.01 * RL))
        biological_age = KSS * RL + 18

    if steps >= 15000:
        biological_age -= 0.5
    elif steps < 10000:
        deficit_percentage = (10000 - steps) / 10000 * 100
        biological_age += 0.5 * (deficit_percentage // 10)
        if steps == 0:
            biological_age += 5

    return round(biological_age, 1)

print(calculate_biological_age(161, 80, 45, 52, 0, False))