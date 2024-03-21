# utils.py

def is_valid_move(pos, size, maze):
    x, y = pos
    if x < 0 or x >= size[1]:
        return False
    if y < 0 or y >= size[0]:
        return False
    if maze[x][y] == 1:
        return False
    return True