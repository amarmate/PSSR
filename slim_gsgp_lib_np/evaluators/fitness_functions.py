# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module provides various error metrics functions for evaluating machine learning models.
"""

import numpy as np 


def rmse(y_true: np.ndarray=None, y_pred: np.ndarray=None, errors=None) -> np.ndarray:
    """
    Compute Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    errors : np.ndarray
        Errors between true and predicted values. If None, it will be computed.

    Returns
    -------
    np.ndarray
        RMSE value.
    """
    if errors is None:
        errors = y_true - y_pred
    return np.sqrt(np.mean(np.square(errors), axis=len(errors.shape) - 1))

def mse(y_true: np.ndarray=None, y_pred: np.ndarray=None, errors=None) -> np.ndarray:
    """
    Compute Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    errors : np.ndarray
        Errors between true and predicted values. If None, it will be computed.

    Returns
    -------
    np.ndarray
        MSE value.
    """
    if errors is None:
        errors = y_true - y_pred
    return np.mean(np.square(errors), axis=len(errors.shape) - 1)

def mae(y_true: np.ndarray, y_pred: np.ndarray, errors=None) -> np.ndarray:
    """
    Compute Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    errors : np.ndarray
        Errors between true and predicted values. If None, it will be computed.

    Returns
    -------
    np.ndarray
        MAE value.
    """
    if errors is None:
         errors = y_true - y_pred
    return np.mean(np.abs(errors), axis=len(errors.shape) - 1)

def mae_int(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute Mean Absolute Error (MAE) for integer values.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    np.ndarray
        MAE value for integer predictions.
    """
    return np.mean(np.abs(y_true - np.round(y_pred)), axis=len(y_pred.shape) - 1)

def signed_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute signed errors between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    np.ndarray
        Signed error values.
    """
    return y_true - y_pred

