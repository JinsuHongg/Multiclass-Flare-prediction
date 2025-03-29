import numpy as np

from typing import Dict, Optional
import numpy as np

class cp:
    def __init__(self, confidence: float = 0.95) -> None:
        """
        Initializes the conformal prediction (cp) class.

        Args:
            confidence (float): Desired confidence level (default: 0.95).
        """
        self.confidence: float = confidence
        self.q_hat: Optional[float] = None  # q_hat is initially None

    def label_q(self, cal_dict: Dict[str, np.ndarray]) -> float:
        """
        Computes the conformal quantile (q_hat) using the calibration set.

        Args:
            cal_dict (dict): Calibration data with 'softmax', 'prediction', and 'label'.
        
        Returns:
            float: The computed q_hat.
        """
        n: int = cal_dict['softmax'].shape[0]
        scores: np.ndarray = 1 - self.find_predprob_truelabel(cal_dict)
        confidence_corrected: float = np.ceil((n + 1) * self.confidence) / n
        self.q_hat = float(np.quantile(scores, confidence_corrected, method='higher'))
        return self.q_hat

    def label_region(self, val_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Computes the prediction region for the validation set.

        Args:
            val_dict (dict): Validation data with 'softmax', 'prediction', and 'label'.
        
        Returns:
            np.ndarray: Binary array indicating the prediction region.
        """
        if self.q_hat is None:
            raise ValueError("q_hat is not computed. Call label_q() before label_region().")

        region: np.ndarray = np.where(val_dict['softmax'] >= (1 - self.q_hat), val_dict['softmax'], 0)
        return region

    def aps_q(self, cal_dict: Dict[str, np.ndarray]) -> float:
        
        # define confidence first
        n: int = cal_dict['softmax'].shape[0]
        confidence_corrected: float = np.ceil((n + 1) * self.confidence) / n

        # 
        Id_sort = np.argsort(cal_dict['softmax'], axis=1)[:, ::-1]
        # cal_arr_sort = np.take_along_axis(cal_dict['softmax'], Id_sort, axis=1).cumsum(axis=1)
        cal_arr_sort = self.find_cumsum_descending(cal_dict, Id_sort)
        arr_scores = np.take_along_axis(cal_arr_sort, Id_sort.argsort(axis=1), axis=1)[np.arange(size_cal), label]
        self.q_hat = float(np.quantile(arr_scores, q = confidence_corrected, method='higher'))

        return self.q_hat
    
    def aps_region(self, val_dict: Dict[str, np.ndarray]) -> np.ndarray:

        """
        Computes the prediction region for the validation set.

        Args:
            val_dict (dict): Validation data with 'softmax', 'prediction', and 'label'.
        
        Returns:
            np.ndarray: Binary array indicating the prediction region.
        """

        if self.q_hat is None:
            raise ValueError("q_hat is not computed. Call label_q() before label_region().")

        Id_sort = val_dict['softmax'].argsort(1)[:, ::-1]
        # val_arr_sort = np.take_along_axis(val_dict['softmax'], Id_sort, axis=1).cumsum(axis=1)
        val_arr_sort = self.find_cumsum_descending(val_dict['softmax'], Id_sort)
        prediction_sets = np.take_along_axis(val_arr_sort <= self.q_hat, Id_sort.argsort(axis=1), axis=1)
        
        # # if APS has zero set, let the set have setsize of 1
        # non_zero_count = np.count_nonzero(prediction_sets, axis=1)
        # block_allzero = np.zeros((arr_result.shape[0], arr_result.shape[1]))
        # for index, bool_val in enumerate(non_zero_count==0):
        #     if bool_val:
        #         max_index = np.argmax(arr_result[index, :])
        #         block_allzero[index, max_index] = np.max(arr_result[index, :])

        return val_dict['softmax'] * prediction_sets

    @staticmethod
    def find_predprob_truelabel(cal_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Finds the probability of the true label for each sample.

        Args:
            cal_dict (dict): Calibration data with 'softmax' and 'label'.
        
        Returns:
            np.ndarray: Probabilities of true labels for each sample.
        """
        return cal_dict['softmax'][np.arange(cal_dict['softmax'].shape[0]), cal_dict['label']]

    @staticmethod
    def find_cumsum_descending(in_dict: Dict[str, np.ndarray], index_arr: np.ndarray) -> np.ndarray:
        return np.take_along_axis(in_dict['softmax'], index_arr, axis=1).cumsum(axis=1)



result = np.load("/workspace/Project/Multiclass-Flare-prediction/results/prediction/Alexnet_202503_train12_test4_CP.npy")
print(result)