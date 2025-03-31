import numpy as np
from typing import Dict, Optional, Tuple


class cp:
    def __init__(self, confidence: float = 0.95) -> None:
        """
        Initializes the conformal prediction (cp) class.

        Args:
            confidence (float): Desired confidence level (default: 0.95).
        """
        self.confidence: float = confidence
        self.q_hat: Optional[float] = None  # q_hat is initially None
        self.q_hat_dict: Optional[dict] = None

    def label_q(self, cal_dict: Dict[str, np.ndarray]) -> float:
        """
        Computes the conformal quantile (q_hat) using the calibration set.

        Args:
            cal_dict (dict): Calibration data with 'softmax', 'prediction', and 'label'.

        Returns:
            float: The computed q_hat.
        """
        scores: np.ndarray = 1 - self.find_predprob_truelabel(
            softmax=cal_dict["softmax"], label=cal_dict["label"]
        )
        # confidence_corrected: float = np.ceil((n + 1) * self.confidence) / n
        # self.q_hat = float(np.quantile(scores, confidence_corrected, method='higher'))
        self.q_hat = self.compute_q_hat(scores=scores, confidence=self.confidence)
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
            raise ValueError(
                "q_hat is not computed. Call label_q() before label_region()."
            )

        region: np.ndarray = np.where(
            val_dict["softmax"] >= (1 - self.q_hat), val_dict["softmax"], 0
        )
        return region

    def aps_q(self, cal_dict: Dict[str, np.ndarray]) -> float:

        # define confidence first
        n: int = cal_dict["softmax"].shape[0]
        arr_scores = self.find_cumsum_descending(softmax=cal_dict["softmax"])[
            np.arange(n), cal_dict["label"].flatten()
        ]
        # self.q_hat = float(np.quantile(arr_scores, q = confidence_corrected, method='higher'))
        self.q_hat = self.compute_q_hat(scores=arr_scores, confidence=self.confidence)
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
            raise ValueError(
                "q_hat is not computed. Call label_q() before label_region()."
            )

        # val_arr_sort = np.take_along_axis(val_dict['softmax'], Id_sort, axis=1).cumsum(axis=1)
        val_arr_sort = self.find_cumsum_descending(softmax=val_dict["softmax"])
        # # if APS has zero set, let the set have setsize of 1
        # non_zero_count = np.count_nonzero(prediction_sets, axis=1)
        # block_allzero = np.zeros((arr_result.shape[0], arr_result.shape[1]))
        # for index, bool_val in enumerate(non_zero_count==0):
        #     if bool_val:
        #         max_index = np.argmax(arr_result[index, :])
        #         block_allzero[index, max_index] = np.max(arr_result[index, :])

        return val_dict["softmax"] * (val_arr_sort <= self.q_hat)

    def mcp_q(self, cal_dict: Dict[str, np.ndarray], type: str = "label") -> float:
        """
        Mondrian conformal prediction
        """

        if type == "label":
            self.q_hat_dict = {"label": []}
            for label in np.unique(cal_dict["label"]):
                label_indices = cal_dict["label"] == label
                label_subset = cal_dict["label"][label_indices]
                softmax_subset = cal_dict["softmax"][label_indices.flatten(), :]
                assert softmax_subset.shape[0] == label_subset.shape[0]

                scores: np.ndarray = 1 - self.find_predprob_truelabel(
                    softmax=softmax_subset, label=label_subset
                )
                # confidence_corrected: float = np.ceil((n + 1) * self.confidence) / n
                self.q_hat_dict["label"].append(
                    self.compute_q_hat(scores=scores, confidence=self.confidence)
                )

        elif type == "aps":
            self.q_hat_dict = {"aps": []}
            for label in np.unique(cal_dict["label"]):
                label_indices = cal_dict["label"] == label
                label_subset = cal_dict["label"][label_indices]
                softmax_subset = cal_dict["softmax"][label_indices.flatten(), :]
                assert softmax_subset.shape[0] == label_subset.shape[0]

                n = softmax_subset.shape[0]
                arr_scores = self.find_cumsum_descending(softmax=softmax_subset)[
                    np.arange(n), label_subset
                ]
                self.q_hat_dict["aps"].append(
                    self.compute_q_hat(scores=arr_scores, confidence=self.confidence)
                )

        else:
            raise ValueError("Invalid score type. Use 'label' or 'aps'.")

    def mcp_region(
        self, val_dict: Dict[str, np.ndarray], type: str = "label"
    ) -> np.ndarray:
        """
        Computes the prediction region for the validation set.

        Args:
            val_dict (dict): Validation data with 'softmax', 'prediction', and 'label'.

        Returns:
            np.ndarray: Binary array indicating the prediction region.
        """
        n_classes = len(np.unique(val_dict["label"]))
        if self.q_hat_dict[type] is None:
            raise ValueError(
                "q_hat_dict is not computed. Call mcp_q() before mcp_region()."
            )
        elif len(self.q_hat_dict[type]) != n_classes:
            raise ValueError(
                "q_hat_dict does not have four components. Please check your input npz file."
            )

        softmax_arr = val_dict["softmax"].copy()  # Avoid modifying the original data

        if type == "label":
            for fl_class, q_hat in enumerate(self.q_hat_dict["label"]):

                pred_id = np.argmax(softmax_arr, axis=1)
                rows = (pred_id == fl_class)

                softmax_arr[rows, :] = np.where(
                    softmax_arr[:, rows] >= (1 - q_hat), softmax_arr[:, rows], 0
                )

                # softmax_arr[:, fl_class] = np.where(
                #     softmax_arr[:, fl_class] >= (1 - q_hat), softmax_arr[:, fl_class], 0
                # )

        elif type == "aps":
            val_arr_sort = self.find_cumsum_descending(
                softmax=val_dict["softmax"]
            )  # Ensure correct method call
            for fl_class, q_hat in enumerate(
                self.q_hat_dict["aps"]
            ):  # Corrected dictionary key
                
                pred_id = np.argmax(val_dict["softmax"], axis=1)
                rows = (pred_id == fl_class)

                softmax_arr[rows, :] = softmax_arr[rows, :] * (
                    val_arr_sort[rows, :] <= q_hat
                )

                # softmax_arr[:, fl_class] = softmax_arr[:, fl_class] * (
                #     val_arr_sort[:, fl_class] <= q_hat
                # )

        return softmax_arr

    @staticmethod
    def find_predprob_truelabel(softmax: np.ndarray, label: np.ndarray) -> np.ndarray:
        """
        Finds the probability of the true label for each sample.

        Args:
            softmax (numpy array): 'softmax' outputs.
            label (numpy array): label data.

        Returns:
            np.ndarray: Probabilities of true labels for each sample.
        """
        return softmax[np.arange(softmax.shape[0]), label.flatten()]

    @staticmethod
    def find_cumsum_descending(softmax: np.ndarray) -> np.ndarray:
        """
        Finds the cumulative sum when the true label for each sample is reached.

        Args:
            softmax (numpy array): softmax outputs from a model.

        Returns:
            np.ndarray: cumulative sum of softmax outcomes for each sample.
        """
        Id_sort = np.argsort(softmax, axis=1)[:, ::-1]
        cum_sum = np.take_along_axis(softmax, Id_sort, axis=1).cumsum(axis=1)
        arr_scores = np.take_along_axis(cum_sum, Id_sort.argsort(axis=1), axis=1)
        return arr_scores

    @staticmethod
    def compute_q_hat(scores: np.ndarray, confidence) -> float:
        confidence_corrected: float = (
            np.ceil((scores.shape[0] + 1) * confidence) / scores.shape[0]
        )
        return float(np.quantile(scores, confidence_corrected, method="higher"))


def coverage_and_length(
    pred_region: np.ndarray, label: np.ndarray
) -> Tuple[float, float]:
    # Ensure input shapes match
    assert label.shape[0] == pred_region.shape[0]

    # Calculate coverage more efficiently by directly indexing
    # This avoids creating the range array and is more readable
    avg_cov = np.mean(pred_region[np.arange(label.shape[0]), label.flatten()] != 0)

    # Calculate length more efficiently
    # Using axis=1 sum once, then taking the mean is faster
    avg_length = np.mean(np.sum(pred_region != 0, axis=1))

    return avg_cov, avg_length
