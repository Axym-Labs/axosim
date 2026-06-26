from __future__ import annotations

# Constants copied from the upstream NeuronIO utilities. Keeping them local avoids
# importing the plotting-heavy data utility module just to run the model.
DEFAULT_Y_TRAIN_SOMA_BIAS = -67.7
DEFAULT_Y_SOMA_THRESHOLD = -55.0
DEFAULT_Y_TRAIN_SOMA_SCALE = 1 / 10

NEURONIO_SIM_PER_FILE = 128
NEURONIO_SIM_LEN = 6000
NEURONIO_DATA_DIM = 1278
NEURONIO_LABEL_DIM = 2
