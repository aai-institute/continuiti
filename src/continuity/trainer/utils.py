from time import time


class Timer:
    """Context manager to measure elapsed time.

    Example:
    ```
    with Timer:
        long_function()
    ```

    """
    def __init__(self, rank: int = 0) -> None:
        self.rank = rank

    def __enter__(self):
        self.start = time()

    def __exit__(self, *args):
        if self.rank == 0:
            print(f"Elapsed time: {time() - self.start:.2f}s")
