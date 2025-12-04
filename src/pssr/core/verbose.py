"""
Verbose output handling for GP evolution.

Provides flexible verbosity modes including periodic printing and progress bars.
"""

import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Try to import tqdm, but make it optional
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None  # type: ignore


class VerboseHandler:
    """
    Manages verbose output for evolution loops.
    
    Supports multiple verbosity modes:
    - 0: Silent (no output)
    - 1: Print every generation (default verbose behavior)
    - N (int > 1): Print every N generations
    - "bar": Show tqdm progress bar
    
    Parameters
    ----------
    verbose : Union[int, str]
        Verbosity mode. Integer for periodic printing, "bar" for progress bar.
    n_generations : int
        Total number of generations for progress bar mode.
    desc : str, optional
        Description for progress bar. Default is "Evolution".
        
    Attributes
    ----------
    mode : str
        Current mode: "silent", "periodic", or "bar"
    interval : int
        Print interval for periodic mode
    pbar : Optional[tqdm]
        Progress bar object if in bar mode
        
    Examples
    --------
    >>> handler = VerboseHandler(verbose=10, n_generations=100)
    >>> for gen in range(100):
    ...     if handler.should_print(gen):
    ...         print(f"Generation {gen}")
    ...     handler.update(gen, {"fitness": 0.5})
    >>> handler.close()
    
    >>> # Progress bar mode
    >>> handler = VerboseHandler(verbose="bar", n_generations=100)
    >>> for gen in range(100):
    ...     handler.update(gen, {"fitness": 0.5})
    >>> handler.close()
    """
    
    def __init__(
        self,
        verbose: Union[int, str],
        n_generations: int,
        desc: str = "Evolution",
    ):
        self.n_generations = n_generations
        self.desc = desc
        self._pbar: Optional[tqdm] = None  # type: ignore
        self._last_metrics: dict = {}
        
        # Parse verbose argument
        if verbose == 0:
            self.mode = "silent"
            self.interval = 0
        elif verbose == "bar":
            if not TQDM_AVAILABLE:
                logger.warning(
                    "tqdm not installed. Falling back to periodic verbose mode. "
                    "Install tqdm with: pip install tqdm"
                )
                self.mode = "periodic"
                self.interval = 1
            else:
                self.mode = "bar"
                self.interval = 0
                self._init_progress_bar()
        elif isinstance(verbose, int) and verbose > 0:
            self.mode = "periodic"
            self.interval = verbose
        else:
            raise ValueError(
                f"Invalid verbose value: {verbose}. "
                "Expected 0 (silent), positive int (interval), or 'bar' (progress bar)."
            )
    
    def _init_progress_bar(self) -> None:
        """Initialize the tqdm progress bar."""
        if TQDM_AVAILABLE and tqdm is not None:
            self._pbar = tqdm(
                total=self.n_generations + 1,  # +1 for generation 0
                desc=self.desc,
                unit="gen",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            )
    
    def should_print(self, generation: int) -> bool:
        """
        Check if this generation should print verbose output.
        
        Parameters
        ----------
        generation : int
            Current generation number.
            
        Returns
        -------
        bool
            True if output should be printed for this generation.
        """
        if self.mode == "silent":
            return False
        elif self.mode == "bar":
            return False  # Bar mode handles display internally
        else:  # periodic
            if self.interval == 1:
                return True
            return generation % self.interval == 0
    
    def update(self, generation: int, metrics: Optional[dict] = None) -> None:
        """
        Update progress for the current generation.
        
        In bar mode, updates the progress bar and displays key metrics.
        In other modes, this is a no-op (printing is handled separately).
        
        Parameters
        ----------
        generation : int
            Current generation number.
        metrics : dict, optional
            Metrics to display. For bar mode, displays selected metrics as postfix.
        """
        if metrics is not None:
            self._last_metrics = metrics
        
        if self.mode == "bar" and self._pbar is not None:
            self._pbar.update(1)
            
            # Update postfix with key metrics
            if metrics:
                postfix = {}
                if "best_fitness" in metrics:
                    postfix["best"] = f"{metrics['best_fitness']:.4f}"
                if "mean_fitness" in metrics:
                    postfix["mean"] = f"{metrics['mean_fitness']:.4f}"
                if "best_test_fitness" in metrics:
                    postfix["test"] = f"{metrics['best_test_fitness']:.4f}"
                
                if postfix:
                    self._pbar.set_postfix(postfix)
    
    def close(self) -> None:
        """
        Close the verbose handler and clean up resources.
        
        Must be called when evolution is complete, especially in bar mode.
        """
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
    
    def __enter__(self) -> "VerboseHandler":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures progress bar is closed."""
        self.close()
    
    @property
    def is_silent(self) -> bool:
        """Check if handler is in silent mode."""
        return self.mode == "silent"
    
    @property
    def is_bar(self) -> bool:
        """Check if handler is in progress bar mode."""
        return self.mode == "bar"
    
    def get_last_metrics(self) -> dict:
        """Get the last metrics passed to update()."""
        return self._last_metrics


def parse_verbose(verbose: Union[int, str]) -> tuple[str, int]:
    """
    Parse verbose argument into mode and interval.
    
    Parameters
    ----------
    verbose : Union[int, str]
        Verbose setting.
        
    Returns
    -------
    tuple[str, int]
        (mode, interval) where mode is "silent", "periodic", or "bar"
    """
    if verbose == 0:
        return "silent", 0
    elif verbose == "bar":
        return "bar", 0
    elif isinstance(verbose, int) and verbose > 0:
        return "periodic", verbose
    else:
        raise ValueError(f"Invalid verbose value: {verbose}")

