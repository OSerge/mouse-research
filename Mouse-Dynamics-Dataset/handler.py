#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Обработчик файлов сессий для обучения нейросети (training_files)
"""

import os
import pandas as pd
import math
import re

root_dir = os.path.abspath(os.path.dirname(__file__))
train_dir = os.path.join(root_dir, "training_files")
score_dir = os.path.join(root_dir, "score")
test_dir = os.path.join(root_dir, "test_files")


def distance(p1: tuple, p2: tuple, p0: tuple) -> float:
    """
    Функция вычисления расстояния произвольной точки (p0) до прямой, соединяющей
    точки p1 и p2 (фактически это отклонение точки от прямой траектории)
    """
    x1, y1 = p1
    x2, y2 = p2
    x0, y0 = p0

    try:
        dist = abs(
            (y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 -y2 * x1
        ) / math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    except ZeroDivisionError:
        dist = math.sqrt((y0 - y1) ** 2 + (x0 - x1) ** 2)
    return dist


walk_train_dir = os.walk(train_dir)
for (dir, _, files) in walk_train_dir:
    os.chdir(dir)
    # Идентификатор юзера из имени дир-и (user[id])
    user_id_regex_obj = re.search(r'\d{1,3}$', dir)
    if not user_id_regex_obj:
        continue
    user_id = int(user_id_regex_obj[0])

    for file in files:
        df = pd.read_csv(file)
        df = df.drop([df.columns[0]], axis=1)
        df = df.rename({'client timestamp': 'timestamp'}, axis=1)

        # Int64Index: индексы строк с меткой 'Move'
        move_idx = df.index[df['state'].isin(['Move'])]

        # Список кортежей (индекс_начала_движения, индекс_конца_движения)
        # По этим индексам будем получать параметры для каждого отдельного
        # движения
        start_end_move_idx = []

        move_idx_length = len(move_idx)

        start_position = move_idx[0]
        end_position = None
        for i in range(1, move_idx_length):
            if (move_idx[i] - move_idx[i - 1]) > 1:
                end_position = move_idx[i - 1]
                start_end_move_idx.append((start_position, end_position))
                start_position = move_idx[i]
            # На последней итерации в любом случае добавить кортеж
            elif i + 1 == move_idx_length:
                end_position = move_idx[i]
                start_end_move_idx.append((start_position, end_position))

        # Создаем новый DataFrame для хранения и вычисления ключевых параметров
        # каждого движения, а затем и усредненных значений для сессии в целом.
        params_df = pd.DataFrame({
            'move_time': [],  # Время движения до остановки
            'move_len': [],  # Длина траектории
            'max_speed': [],  # Макс. скорость движения
            # 'alpha': [],  # Угол начального направления движения (отклонения)
            'sigma': [],  # Сред. квадрат. отклон. реал. траектории от прямой
            'user_id': [],  # ID пользователя
        })
        for start, end in start_end_move_idx:
            temp_df = df.loc[start:end]
            point_amount = len(temp_df)  # кол-во точек в движении
            if point_amount < 3:
                continue
            move_time = round(
                temp_df.iloc[-1]['timestamp'] - temp_df.iloc[0]['timestamp'], 3
            )
            # Список кортежей для расчета ключевых параметров
            time_x_y = [row for row in zip(temp_df.timestamp, temp_df.x, temp_df.y)]
            max_speed = 0
            move_len = 0
            sigma = 0
            start_point = time_x_y[0][1:]  # Первая точка движения (x, y)
            end_point = time_x_y[-1][1:]  # Последняя точка движения
            point_deviations = []
            for i in range(1, point_amount):
                # Расчет макс. скорости
                L_loc = math.sqrt(
                    (time_x_y[i][1] - time_x_y[i - 1][1]) ** 2 +
                    (time_x_y[i][2] - time_x_y[i - 1][2]) ** 2
                )
                T_loc = time_x_y[i][0] - time_x_y[i - 1][0]
                try:
                    V_loc = L_loc / T_loc
                except ZeroDivisionError as e:
                    V_loc = L_loc / 0.001
                if V_loc > max_speed:
                    max_speed = round(V_loc, 3)

                # Расчет общей длины траектории
                move_len += L_loc

                # Расчет отклонений (расстояния) точек от прямой
                mid_point = time_x_y[i][1:]  # Промежуточная точка (x, y)
                deviation = distance(start_point, end_point, mid_point)
                point_deviations.append(deviation)

            # Расчет углов наклона прямых, проход. через начальную точку и 3 точку траектории
            try:
                p3_p0_angle = math.degrees(math.atan(
                    (time_x_y[2][2] - time_x_y[0][2])    # (y3 - y0)
                    / (time_x_y[2][1] - time_x_y[0][1])  # / (x3 - x0)
                ))
            except ZeroDivisionError as e:
                # Если знаменатель равен 0, значит угол 90 градусов со знаком числителя
                p3_p0_angle = math.copysign(90, time_x_y[2][2] - time_x_y[0][2])

            # ... через начальную и конечную точки
            try:
                pn_p0_angle = math.degrees(math.atan(
                    (time_x_y[-1][2] - time_x_y[0][2])  # (yn - y0)
                    / (time_x_y[-1][1] - time_x_y[0][1])  # / (xn - x0)
                ))
            except ZeroDivisionError as e:
                pn_p0_angle = math.copysign(90, time_x_y[-1][2] - time_x_y[0][2])

            # Угол отклонения траектории -- фактически угол отклонения 3-й точки траектории
            # от прямой, соединяющей начальную и конечную точки. (в градусах)
            alpha = p3_p0_angle - pn_p0_angle

            # Среднеквадратичное отклонение точек от прямой траектории
            sigma = sum([x ** 2 for x in point_deviations]) / len(point_deviations)

            params_df = params_df.append({
                'move_time': move_time,
                'move_len': move_len,
                'max_speed': max_speed,
                # 'alpha': alpha,
                'sigma': sigma,
                'user_id': user_id,
            }, ignore_index=True)

        # Рассчитываем средние арифм. всех параметров по всем движениям в сессии -
        # это и есть курсорчный "почерк" пользователя
        mean_list = params_df.mean().to_list()
        mean_string = ",".join(map(str, [round(m, 3) for m in mean_list])) + "\n"

        user_score_file = os.path.join(score_dir, f"{os.path.basename(dir)}.csv")
        common_score_file = os.path.join(score_dir, f"score.csv")
        with open(user_score_file, "a") as score:
            score.write(mean_string)
        with open(common_score_file, "a") as score:
            score.write(mean_string)
        print(f"Сессия {file} обработана в: \n\t-> {user_score_file} \n\t-> {common_score_file}")

