import pandas as pd
import pytask

import epp_final_project_sbp.data_management.clean_data as cd
import epp_final_project_sbp.data_management.web_scraper as ws
from epp_final_project_sbp.config import SRC


@pytask.mark.produces(SRC / "data" / "data_raw.csv")
def task_webscraping(produces):
    data = pd.DataFrame()
    data_sources = {}
    information = {
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
        "Leagues": {
            "PL": {
                "Foldername": "PL_data",
                "Leaguetag": "PL",
                "Leaguename": "E0",
                "Leagueurl": "https://www.football-data.co.uk/englandm.php",
            },
            "BL": {
                "Foldername": "BL_data",
                "Leaguetag": "BL",
                "Leaguename": "D1",
                "Leagueurl": "https://www.football-data.co.uk/germanym.php",
            },
            "PD": {
                "Foldername": "PD_data",
                "Leaguetag": "PD",
                "Leaguename": "SP1",
                "Leagueurl": "https://www.football-data.co.uk/spainm.php",
            },
            "SA": {
                "Foldername": "SA_data",
                "Leaguetag": "SA",
                "Leaguename": "I1",
                "Leagueurl": "https://www.football-data.co.uk/italym.php",
            },
        },
    }

    for league in information["Leagues"]:
        data_sources[league] = ws.download_data(
            url=information["Leagues"][league]["Leagueurl"],
            years=information["years"],
            beginning_url=information["beginning_url"],
            league=information["Leagues"][league]["Leaguename"],
        )
        data = cd.rbind_list_of_dataframes(data_sources=data_sources[league], data=data)
    data = cd.delete_rows_with_just_nans(df=data)
    data.to_csv(produces, index=False)
