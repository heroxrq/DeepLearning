import urllib3
from bs4 import BeautifulSoup
import re

BASE_URL = r"http://www.dianping.com"
ENTRANCE_URL = r"http://www.dianping.com/phuket/food"

http = urllib3.PoolManager()
response = http.request('GET', ENTRANCE_URL, timeout=5.0, retries=10)

# print(response.status)
# print(response.data)


soup = BeautifulSoup(response.data, 'html.parser')
for link in soup.find_all('a', attrs={"class": "BL"}, limit=15):
    href = link.get('href')
    title = link.get('title')
    print(href)
    print(title)
    shop_url = BASE_URL + href
    print(shop_url)
