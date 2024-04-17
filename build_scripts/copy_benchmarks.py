import logging
import os
from pathlib import Path

import mkdocs.plugins

logger = logging.getLogger(__name__)

root_dir = Path(__file__).parent.parent
benchmarks_dir = root_dir / "benchmarks" / "results"
docs_benchmarks_dir = root_dir / "docs" / "benchmarks"


@mkdocs.plugins.event_priority(100)
def on_pre_build(config):
    logger.info("Temporarily copying benchmark results to docs directory")
    docs_benchmarks_dir.mkdir(parents=True, exist_ok=True)
    filepaths = list(benchmarks_dir.glob("*.json"))

    for file in filepaths:
        target_filepath = docs_benchmarks_dir / file.name

        try:
            if os.path.getmtime(file) <= os.path.getmtime(target_filepath):
                logger.info(f"File '{os.fspath(file)}' hasn't been updated, skipping.")
                continue
        except FileNotFoundError:
            pass
        logger.info(
            f"Creating symbolic link for '{os.fspath(file)}' "
            f"at '{os.fspath(target_filepath)}'"
        )
        target_filepath.symlink_to(file)

    logger.info("Finished copying notebooks to examples directory")


@mkdocs.plugins.event_priority(-100)
def on_shutdown():
    logger.info("Removing temporary examples directory")
    for file in docs_benchmarks_dir.glob("*.json"):
        file.unlink()
