import pandas as pd
import pytask

import epp_final_project_sbp.data_management.clean_data as cd
import epp_final_project_sbp.web_scraping.web_scraper as ws
from epp_final_project_sbp.config import INFORMATION_SCRAPING, SRC


@pytask.mark.produces(SRC / "data" / "data_raw.csv")
def task_webscraping(produces):
    data = pd.DataFrame()
    data_sources = {}
    information = INFORMATION_SCRAPING

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
