import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


def download_data(url, years, beginning_url, league):
    """
    Generates a list of download links for the csv files of the league and years specified.
    After that it downloads the csv files.
    Parameters
    ----------
    url : str
        The url of the website to scrape.
    years : list
        A list of the years to scrape.
    beginning_url : str
        The beginning of the url to add to the scraped links.
    league : str
        The league to scrape.
    Output
    ------
    download_links : list
        A list of the download links for the csv files.
    """
    op = webdriver.ChromeOptions()
    op.add_argument("headless")
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=op)
    driver.get(url)
    html = driver.page_source
    driver.close()
    soup = BeautifulSoup(html)
    list(soup.find("a"))
    download_links = []

    for a in soup.find_all("a", href=True):
        if (
            a["href"].startswith("mmz")
            and any([x in a["href"] for x in years])
            and a["href"].endswith(league + ".csv")
        ):
            download_links.append(a["href"])
    download_links = [beginning_url + x for x in download_links]
    data = __download_csvs(download_links=download_links, league=league, years=years)
    return data


def __download_csvs(download_links, league, years):
    """
    Downloads the csv files from the download links.
    Parameters
    ----------
    download_links : list
        A list of the download links for the csv files.
    league : str
        The league to scrape.
    directory_name : str
        The name of the directory to save the csv files to.
    Output
    ------
    None.
    """
    data = {}
    for link in download_links:
        for year in years:
            if year in link:
                key_year = year
                name = league + key_year
                data[name] = pd.read_csv(link)
    return data
