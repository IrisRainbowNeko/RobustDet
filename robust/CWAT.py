from .base import AdvDataGenerator

class CWAT(AdvDataGenerator):
    def __init__(self, attacker, criterion):
        super().__init__(attacker, loss=criterion)