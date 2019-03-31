class FoodObj:
    def __init__(self, coordinates, type=1, reward=1):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.type = type
        self.is_collected = False
        self.reward = reward

    def eat(self):
        self.is_collected = True
        return self.reward

    def respawn(self):
        self.is_collected = False
