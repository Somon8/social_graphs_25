import urllib
def from_github(path):
    base_url = "https://raw.githubusercontent.com/Somon8/social_graphs_25/main/final-project"
    url_to_file = base_url + path
    return urllib.request.urlopen(url_to_file)


#Example:
# from github_helper import from_github
# df_votes = pd.read_csv(from_github("/voting-data/df_votes_all_periods.csv"))