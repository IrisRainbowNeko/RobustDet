
class Attacker:
    def __init__(self, model, img_transform=(lambda x:x, lambda x:x)):
        self.model = model  # 必须是pytorch的model
        '''self.model.eval()
        for k, v in self.model.named_parameters():
            v.requires_grad = False'''
        self.img_transform=img_transform
        self.forward = lambda attacker, images, labels: attacker.step(images, labels, attacker.loss)

    def set_para(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k,v)

    def set_forward(self, forward):
        self.forward=forward

    def step(self, images, labels, loss):
        pass

    def set_loss(self, loss):
        self.loss=loss

    def attack(self, images, labels):
        pass