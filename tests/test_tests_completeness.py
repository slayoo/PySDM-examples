import pathlib

import pytest
from conftest import TEST_SUITES, findfiles, get_selected_test_suites


def test_all_cases_in_testsuites():
    """raise error, e.g., if a newly added example is not within TEST_SUITES dict"""
    tmp = findfiles(
        pathlib.Path(__file__).parent.parent.absolute().joinpath("PySDM_examples"),
        r".*\.(py|ipynb)$",
    )
    all_files = list(
        filter(
            lambda x: pathlib.Path(x).name != "__init__.py"
            and "utils" not in pathlib.Path(x).parts,
            tmp,
        )
    )

    selected_paths_set = set()
    for suite_name in TEST_SUITES:
        selected_paths_set.update(
            map(str, get_selected_test_suites(suite_name, all_files))
        )

    assert selected_paths_set == set(all_files)


def test_no_cases_in_multiple_testsuites():
    """raise an error if an example is featured in multiple TEST_SUITES"""
    flattened_suites = sum(list(TEST_SUITES.values()), [])

    assert len(set(flattened_suites)) == len(flattened_suites)


@pytest.fixture(
    params=findfiles(
        pathlib.Path(__file__)
        .parent.parent.absolute()
        .joinpath("PySDM_examples")
        .joinpath("utils"),
        r".*\.(py)$",
    )
)
def utils_filename(request):
    return request.param


# pylint: disable=redefined-outer-name
def test_run_utils(utils_filename):
    with open(utils_filename, encoding="utf8") as f:
        exec(f.read(), {"__name__": "__main__"})  # pylint: disable=exec-used
