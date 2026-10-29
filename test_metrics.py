import numpy as np
from src.metrics import binary_metrics

def test_perfect():
    x=np.array([0,1,1,0]); m=binary_metrics(x,x)
    assert abs(m["dice"]-1)<1e-7 and abs(m["iou"]-1)<1e-7

def test_no_overlap():
    m=binary_metrics(np.array([1,1,0,0]),np.array([0,0,1,1]))
    assert m["dice"]<1e-7 and m["iou"]<1e-7

def test_both_empty_sets_score_one():
    m=binary_metrics(np.zeros(4,dtype=bool),np.zeros(4,dtype=bool))
    assert all(value == 1 for value in m.values())

def test_no_prediction_for_positive_target_has_zero_precision_and_recall():
    m=binary_metrics(np.zeros(4,dtype=bool),np.array([1,0,0,0],dtype=bool))
    assert m["precision"] == 0
    assert m["recall_sensitivity"] == 0

def test_false_positive_for_empty_target_has_zero_precision_and_recall():
    m=binary_metrics(np.array([1,0,0,0],dtype=bool),np.zeros(4,dtype=bool))
    assert m["precision"] == 0
    assert m["recall_sensitivity"] == 0
