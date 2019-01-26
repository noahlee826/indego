import requests, zipfile, io, os

# UNFINISHED

URLS_FILE = 'indego-tripdata-urls.txt'

# Gather the urls from URLS_FILE into an array
urls_list = []
with open(URLS_FILE) as f:
    for line in f:
        urls_list.append(line.rstrip())
#print(urls_list)

#Download the zips
csvs = []
for url in urls_list:
    response = requests.get(url)
    zipped = zipfile.ZipFile(io.BytesIO(response.content))
    zipped.extractall()

cwd = os.getcwd()