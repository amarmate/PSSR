import numpy as np 
import numpy.typing as npt

from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import StandardScaler

class NormalizationMixin(BaseEstimator):
    def __init__(self, normalize: bool = True, X_scaler=None, y_scaler=None) -> None:
        self.normalize = normalize
        self.X_scaler = X_scaler if X_scaler is not None else StandardScaler()
        self.y_scaler = y_scaler if y_scaler is not None else StandardScaler()
        
    def _fit_normalize(self, X: npt.ArrayLike, y: npt.ArrayLike) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.normalize:
            self.X_scaler_ = clone(self.X_scaler).fit(X)
            self.y_scaler_ = clone(self.y_scaler).fit(y.reshape(-1, 1))
            X = self.X_scaler_.transform(X)
            y = self.y_scaler_.transform(y.reshape(-1, 1)).ravel()
        else:
            self.X_scaler_ = None
            self.y_scaler_ = None
        return X, y
    
    def _transform_X(self, X: npt.ArrayLike) -> np.ndarray:
        X = np.asarray(X)
        if getattr(self, "X_scaler_", None) is not None:
            assert self.X_scaler_ is not None 
            X = self.X_scaler_.transform(X)
        return np.asarray(X)

    def _inverse_transform_y(self, y_pred: npt.ArrayLike) -> np.ndarray:
        y_pred = np.asarray(y_pred).reshape(-1, 1)
        if getattr(self, "y_scaler_", None) is not None:
            assert self.y_scaler_ is not None
            y_pred = self.y_scaler_.inverse_transform(y_pred)
        return np.ravel(y_pred)

