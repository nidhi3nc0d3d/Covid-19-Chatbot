# import urllib library
from urllib.request import urlopen
from datetime import date
# import json
import json

# store the URL in url as
# parameter for urlopen
url = "https://api.apify.com/v2/key-value-stores/toDWvRj1JpTXiM8FF/records/LATEST?disableRedirect=true"

# store the response of URL
response = urlopen(url)

# storing the JSON response
# from url in data
data_json = json.loads(response.read())

# storing json response
x = date.today()
date = x.strftime("%d / %m / %Y")
total_cases = data_json['totalCases']
active_cases = data_json['activeCases']
recovered_cases = data_json['recovered']
deaths = data_json['deaths']


# print the json response
def all_cases():
    print("Bot: ")
    print("-" * 15 + str("COVID-19 CASES") + "-" * 15)
    print("Today's date   : " + str(date))
    print("Total Cases    : " + str(total_cases))
    print("Active Cases   : " + str(active_cases))
    print("Recovered Cases: " + str(recovered_cases))
    print("Death Cases    : " + str(deaths))


def total():
    print("Bot: ")
    print("Today's date   : " + str(date))
    print("Total Cases of Covid-19 in India is " + str(total_cases))


def active():
    print("Bot: ")
    print("Today's date   : " + str(date))
    print("Number of Active Cases of Covid-19 in India is " + str(active_cases))


def recover():
    print("Bot: ")
    print("Today's date   : " + str(date))
    print("Number of Recovered Cases of Covid-19 in India is " + str(recovered_cases))
    return None


def death_cases():
    print("Bot: ")
    print("Today's date   : " + str(date))
    print("Number of Death Cases of Covid-19 in India is " + str(deaths))
