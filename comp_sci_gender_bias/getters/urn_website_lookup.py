from comp_sci_gender_bias import PROJECT_DIR
import pandas as pd

URN_WEBSITE_LOOKUP_PATH = (
    PROJECT_DIR
    / "inputs/data/urn_school_website_lookup/get_information_schools_urn_website.csv"
)


def urn_website_lookup():
    """Load dataframe containing school websites and
    related school unique reference number
    Data source: https://www.get-information-schools.service.gov.uk/Search?SelectedTab=Establishments
    """
    return pd.read_csv(URN_WEBSITE_LOOKUP_PATH)
