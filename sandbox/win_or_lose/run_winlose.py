from pathlib import Path

from winlose_sklearn import train_logreg
from winlose_torch import ClassNet, train_torch

from meml_griz import snifcheck

PWD = Path(__file__).parent


def main():

    # Train SKlearn logistric regression
    train_logreg()

    # Train torch model
    train_torch()

    # Compare the classification reports for both models
    sniffs = {
        "sklearn": snifcheck.ClassifierSniff.from_serialized_model("winlose_sklearn"),
        "torch": snifcheck.ClassifierSniff.from_serialized_model("winlose_torch"),
    }

    for name, check in sniffs.items():
        print("*" * 32)
        print(f"Method: {name:<10s}")
        print(check.classification_report)


if __name__ == "__main__":
    main()
