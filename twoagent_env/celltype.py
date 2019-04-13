class CellType(object):
    EMPTY = " "
    PLAYER = "@"
    APPLE = "$"
    WALL = "#"
    OPPONENT = "O"
    AGENT_FRONT = "x"
    BEAM = "*"

class Colors:
    SCREEN_BACKGROUND = (10, 10, 10)  # BLACK
    CELL_TYPE = {
        CellType.WALL: (125, 125, 125),  # GRAY
        CellType.AGENT_FRONT: (50, 50, 50),  # DARK GRAY
        CellType.PLAYER: (0, 0, 255),  # BLUE
        CellType.OPPONENT: (255, 0, 0),  # RED
        CellType.APPLE: (0, 255, 0),  # GREEN
        CellType.BEAM: (255, 204, 51),  # Yellow
    }
