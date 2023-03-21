"""Tests for the regression model."""

import pytest
from epp_final_project_sbp.web_scraping.web_scraper import __get_soup_file


@pytest.fixture()
def download_data_typical_information():
    return {
        "beginning_url": "https://www.football-data.co.uk/",
        "years": [
            "2223",
            "2122",
            "2021",
            "1920",
            "1819",
            "1718",
            "1617",
            "1516",
            "1415",
            "1314",
            "1213",
            "1213",
            "1112",
        ],
        "PL": {
            "Foldername": "PL_data",
            "Leaguetag": "PL",
            "Leaguename": "E0",
            "Leagueurl": "https://www.football-data.co.uk/englandm.php",
        },
    }


def test_soup_function_valid_url(download_data_typical_information):
    """Test if the soup function returns a soup file."""
    url = download_data_typical_information["PL"]["Leagueurl"]
    soup = __get_soup_file(url=url)
    assert soup is not None


def test_soup_invalid_url(download_data_typical_information):
    """Test if the soup function returns None if the url is invalid."""
    download_data_typical_information["PL"][
        "Leagueurl"
    ] = "https://www.football-data.co.uk/englandm"
    url = download_data_typical_information["PL"]["Leagueurl"]
    soup = __get_soup_file(url=url)
    with pytest.raises(AssertionError):
        assert soup is None
