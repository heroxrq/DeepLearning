import heapq
import math


def calculate_angle(barcode_coordinate):
    point_distance_info_list = []
    for i in range(0, 8, 2):
        point_a_x = barcode_coordinate[i % 8]
        point_a_y = barcode_coordinate[(i+1) % 8]
        point_b_x = barcode_coordinate[(i+2) % 8]
        point_b_y = barcode_coordinate[(i+3) % 8]

        distance = calculate_distance(point_a_x, point_a_y, point_b_x, point_b_y)

        point_distance_info = (point_a_x, point_a_y, point_b_x, point_b_y, distance)
        point_distance_info_list.append(point_distance_info)

    # find the top 2 longest edge
    top_2_distance_point_info = heapq.nlargest(2, point_distance_info_list, key=lambda x: x[-1])

    delta_x_1 = top_2_distance_point_info[0][2] - top_2_distance_point_info[0][0]
    delta_x_2 = top_2_distance_point_info[1][2] - top_2_distance_point_info[1][0]

    angle1 = math.acos(math.fabs(delta_x_1) / top_2_distance_point_info[0][-1])
    angle2 = math.acos(math.fabs(delta_x_2) / top_2_distance_point_info[1][-1])

    avg_angle = (angle1+angle2)*180/(2*math.pi)

    top_1_distance_point_info = [(top_2_distance_point_info[0][0], top_2_distance_point_info[0][1]),
                                 (top_2_distance_point_info[0][2], top_2_distance_point_info[0][3])]

    min_x_point = min(top_1_distance_point_info, key=lambda x: x[0])

    flag = 0
    for point in top_1_distance_point_info:
        if point[1] < min_x_point[1]:
            flag = 1

    return avg_angle if flag else 180 - avg_angle


def calculate_distance(point_a_x, point_a_y, point_b_x, point_b_y):
    delta_x = point_a_x - point_b_x
    delta_y = point_a_y - point_b_y

    distance = math.sqrt(delta_x * delta_x + delta_y * delta_y)
    return distance


if __name__ == '__main__':
    barcode_coordinate_ = [int(x) for x in "2737 2053 3127 2279 3649 1651 3261 1388".split(" ")]
    print(calculate_angle(barcode_coordinate_))
