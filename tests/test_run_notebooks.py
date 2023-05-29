# pylint: disable=wrong-import-position
# https://bugs.python.org/issue37373
import sys

if sys.platform == "win32" and sys.version_info[:2] >= (3, 7):
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def test_run_notebooks(notebook_filename, tmp_path):
    print(notebook_filename)
    with open(notebook_filename, encoding="utf8") as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=15 * 60, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": tmp_path}})


def test_all_cases_in_testsuites():
    # raise error, e.g., if a newly added example is not within TEST_SUITES dict
    pass


def test_no_cases_in_multiple_testsuites():
    # raise an error if an example is featured in multiple TEST_SUITES
    pass
