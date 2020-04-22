from baseline.model import Baseline
from data import select_columns
from evaluate import evaluate


def run():
    data = select_columns()
    model = Baseline()

    evaluate(data, model)
