
class CleanGenerator:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, data, label):
        return data

class BaseAdvDataGenerator:
    def __init__(self, attacker):
        self.attacker=attacker

    def __iter__(self):
        return iter(self.attacker)

    def generate(self, data, label):
        return self.attacker.attack(data, label)

class AdvDataGenerator(BaseAdvDataGenerator):
    def __init__(self, attacker, loss):
        super().__init__(attacker)
        self.attacker.set_loss(loss=loss)

class CLS_ADG(AdvDataGenerator):
    def __init__(self, attacker, criterion):
        super().__init__(attacker, loss=lambda out, la: criterion(out, la)[1])

class LOC_ADG(AdvDataGenerator):
    def __init__(self, attacker, criterion):
        super().__init__(attacker, loss=lambda out, la: criterion(out, la)[0])

class CON_ADG(AdvDataGenerator):
    def __init__(self, attacker, criterion, rate=[1,1]):
        def add(res, rate):
            return res[0]*rate[0]+res[1]*rate[1]
        super().__init__(attacker, loss=lambda out, la: add(criterion(out, la), rate))