import os
import pathlib
import re

from conftest import TEST_SUITES, findfiles, get_selected_test_suites


def test_all_cases_in_testsuites():
    """raise error, e.g., if a newly added example is not within TEST_SUITES dict"""
    all_files = findfiles(
        pathlib.Path(__file__).parent.parent.absolute(), r".*\.(py|ipynb)$"
    )
    selected_paths_set = set()
    for suite_name in TEST_SUITES.keys():
        selected_paths_set.update(get_selected_test_suites(suite_name, all_files))

    assert selected_paths_set == set(all_files)


def test_no_cases_in_multiple_testsuites():
    """raise an error if an example is featured in multiple TEST_SUITES"""
    pass
