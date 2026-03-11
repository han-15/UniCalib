from enum import Enum

class EngineMode(Enum):
    """
    Enum class representing the different types of data splits.

    Attributes:
        TRAIN (int): Represents the training data split.
        VALID (int): Represents the validation data split.
        TEST (int): Represents the test data split.
    """
    TRAIN = 0
    VALID = 1
    TEST = 2