from torch import nn
import torch

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

class CNN(nn.Module):
    def __init__(self, nx: int, ny: int, na: int):
        """À MODIFIER QUAND NÉCESSAIRE.
        Ce constructeur crée une instance de réseau de neurones convolutif (CNN).
        L'architecture choisie doit être choisie de façon à capter toute la complexité du problème
        sans pour autant devenir intraitable (trop de paramètres d'apprentissages).

        :param na: Le nombre d'actions
        :type na: int
        """
        super(CNN, self).__init__()
        self.to(device)

        n_neurons = 64

        self.layers = nn.Sequential(
            nn.Conv2d(1, n_neurons, 9, stride=1, padding="same", bias=True).to(device),
            nn.ReLU().to(device),
            nn.Conv2d(n_neurons, n_neurons, 5, stride=1, padding="same", bias=True).to(device),
            nn.ReLU().to(device),
             nn.Conv2d(n_neurons, n_neurons, 3, stride=1, padding="same", bias=True).to(device),
            nn.ReLU().to(device),
            nn.Flatten().to(device),
            nn.Linear(ny * nx * n_neurons, 120).to(device),
            nn.Linear(120, na).to(device)
        ).to(device)

        self.layers.apply(weights_init).to(device)

    def forward(self, x):
        """Cette fonction réalise un passage dans le réseau de neurones.

        :param x: L'état
        :return: Le vecteur de valeurs d'actions (une valeur par action)
        """
        qvalues = self.layers(x)
        return qvalues


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight).to(device)
        nn.init.zeros_(m.bias).to(device)
