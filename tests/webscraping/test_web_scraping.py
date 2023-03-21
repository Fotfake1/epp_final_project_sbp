"""Tests for the regression model."""

import pytest
from epp_final_project_sbp.web_scraping.web_scraper import (
    __get_soup_file,
    download_data,
)


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


def test_download_data_valid_input(download_data_typical_information):
    """test, if download links subfunction raises and error with wrong starting url.

    test, if the function runs, when inserting a wrong starting url, but the correct one
    is in the download links.

    """
    data = download_data(
        url=download_data_typical_information["PL"]["Leagueurl"],
        years=download_data_typical_information["years"],
        beginning_url=download_data_typical_information["beginning_url"],
        league=download_data_typical_information["PL"]["Leaguename"],
    )
    assert isinstance(data, dict)
    assert len(data) > 0


def test_download_data_invalid_starting_url(download_data_typical_information):
    """Test if the download_data function raises an error, if the URL is invalid."""
    download_data_typical_information["PL"][
        "Leagueurl"
    ] = "https://www.football-datafalse.co.uk/englandm"
    url = download_data_typical_information["PL"]["Leagueurl"]

    with pytest.raises(Exception) as e:
        download_data(
            url=url,
            years=download_data_typical_information["years"],
            beginning_url=download_data_typical_information["beginning_url"],
            league=download_data_typical_information["PL"]["Leaguename"],
        )
    assert "The url is invalid. Could not get the soup file." in str(e.value)


def test_download_data_invalid_year_formats(download_data_typical_information):
    """Test if the download_data function raises an error, if the URL is invalid."""
    download_data_typical_information["years"] = [2020, 2021]

    with pytest.raises(AssertionError) as e:
        download_data(
            url=download_data_typical_information["PL"]["Leagueurl"],
            years=download_data_typical_information["years"],
            beginning_url=download_data_typical_information["beginning_url"],
            league=download_data_typical_information["PL"]["Leaguename"],
        )
    assert str(e.value) == "The year list must contain only strings."


def test_download_data_invalid_years(download_data_typical_information):
    """Test if the download_data function raises an error, if the URL is invalid."""
    download_data_typical_information["years"] = ["1966", "1955"]

    with pytest.raises(AssertionError) as e:
        download_data(
            url=download_data_typical_information["PL"]["Leagueurl"],
            years=download_data_typical_information["years"],
            beginning_url=download_data_typical_information["beginning_url"],
            league=download_data_typical_information["PL"]["Leaguename"],
        )

    assert (
        str(e.value)
        == "Download list is empty. This means, that on the url provided, there are no .csv files, which met the download cirteria for years, league and how the url should be structured."
    )


def test_soup_invalid_url(download_data_typical_information):
    """Test if the soup function returns None if the url is invalid."""
    download_data_typical_information["PL"][
        "Leagueurl"
    ] = "https://www.football-data.co.uk/englandm"
    url = download_data_typical_information["PL"]["Leagueurl"]
    soup = __get_soup_file(url=url)
    with pytest.raises(AssertionError):
        assert soup is None
