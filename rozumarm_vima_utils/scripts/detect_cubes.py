def mock_detect_cubes():
    obj_posquats = [
        ((0, 0), (0, 0, 0, 1)),
        ((0, 0), (0, 0, 0, 1))
    ]
    return obj_posquats


if __name__ == '__main__':
    pq = mock_detect_cubes()
    print(pq)