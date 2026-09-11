from ml.evaluation.metrics import CostConfig, binary_metrics, decision_cost


def test_precision_recall_sanity():
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.8, 0.9]
    m = binary_metrics(y, p, 0.5)
    assert m["precision"] == 1
    assert m["recall"] == 1
    assert m["fp"] == 0


def test_false_positive_cost_is_counted():
    y = [0, 0]
    p = [0.9, 0.9]
    costs = CostConfig(120, 80, 220, 0.28, 0.55)
    c = decision_cost(y, p, 0.5, [1000, 1000], costs)
    assert c["fp_cost_total"] == 240
    assert c["fn_cost_total"] == 0
