import numpy as np

# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import f1_score


def TSS(table):
    """
    Calculates the true skill score based on the true classes and the predicted ones.

    Note: Keep in mind that the order of the class labels 'labels' in
    'confusion_matrix' defines the positive and negative classes. Here we set it to
    ["CBN", "XM"].

    (From Bobra's paper) - The flaring ARs correctly predicted as flaring are called true
    positives (TP), the flaring ARs incorrectly predicted as non-flaring are false negatives (FN),
    the non-flaring ARs correctly predicted as non-flaring are true negatives (TN), and the
    non-flaring ARs incorrectly predicted as flaring are false positives (FP). From these four
    quantities, various metrics are computed.
    """
    # this order is in line with the confusion_matrix function we use here.
    TN, FP, FN, TP = table

    tp_rate = TP / float(TP + FN) if TP > 0 else 0  # also known as "recall"
    fp_rate = FP / float(FP + TN) if FP > 0 else 0
    return tp_rate - fp_rate


def HSS1(table):
    """
    Calculates the Heidke skill score based on the output of the confusion table function
    This is the first way to compute - using the Barnes & Leka (2008) definition
    """
    TN, FP, FN, TP = table
    N = TN + FP
    P = TP + FN
    HSS = (TP + TN - N) / float(P)
    return HSS


def HSS2(table):
    """
    Calculates the Heidke skill score based on the output of the confusion table function
    This is the second way to compute - using the Mason & Hoeksema (2010) definition
    """
    TN, FP, FN, TP = table
    N = TN + FP
    P = TP + FN
    HSS = (2 * (TP * TN - FN * FP)) / float((P * (FN + TN) + (TP + FP) * N))
    return HSS


def GSS(table):
    # this order is in line with the confusion_matrix function we use here.
    TN, FP, FN, TP = table

    CH = ((TP + FP) * (TP + FN)) / (TP + FP + FN + TN)
    GSS = (TP - CH) / (TP + FP + FN - CH)
    return GSS


def precisionPos(table):
    TN, FP, FN, TP = table
    precisionPos = TP / float(TP + FP)
    return precisionPos


def TPR(table):
    TN, FP, FN, TP = table
    TPR = TP / float(TP + FN)
    return TPR


def precisionNeg(table):
    TN, FP, FN, TP = table
    precisionNeg = TN / float(TN + FN)
    return precisionNeg


def TNR(table):
    TN, FP, FN, TP = table
    TNR = TN / float(TN + FP)
    return TNR


def FAR(table):
    TN, FP, FN, TP = table
    FAR = FP / float(TP + FP)
    return FAR


def POFD(table):
    TN, FP, FN, TP = table
    POFD = FP / float(TN + FP)
    return POFD


def F1Pos(table):
    TN, FP, FN, TP = table
    precision = TP / float(TP + FP)
    recall = TP / float(TP + FN)
    f1 = 2 * ((precision * recall) / (precision + recall))
    return f1


def F1Neg(table):
    TN, FP, FN, TP = table
    precision = TN / float(TN + FN)
    recall = TN / float(TN + FP)
    f1 = 2 * ((precision * recall) / (precision + recall))
    return f1


# table should be np array multi-class confusion matrix
# from sklearn.metrics import confusion_matrix
# range: -inf~1
def HSS_multiclass(table):
    N = table.sum()
    TP_sum = np.diag(table).sum() / (N)
    Forcast_times_obs = (table.sum(axis=0) * table.sum(axis=1)).sum() / (N**2)

    return (TP_sum - Forcast_times_obs) / (1 - Forcast_times_obs)


# range: -1~1
def TSS_multiclass(table):
    N = table.sum()
    TP_sum = np.diag(table).sum() / (N)
    Forcast_times_obs = (table.sum(axis=0) * table.sum(axis=1)).sum() / (N**2)
    obs_sqr = (table.sum(axis=1) ** 2).sum() / (N**2)

    return (TP_sum - Forcast_times_obs) / (1 - obs_sqr)
