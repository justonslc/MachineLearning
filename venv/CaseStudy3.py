#Juston ELlis
#Case Study 3 
#Movie html parser

import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import csv

# GET request to the website
url = "https://www.timeout.com/film/best-movies-of-all-time"
response = requests.get(url).text

# bsoup parses the html for elements
soup = BeautifulSoup(response, "html.parser")

# finding elements in webpage
movie_elements = soup.find_all("h3", class_="_h3_cuogz_1")

if movie_elements:
    # Create a list to store movie data
    movies = []

    for movie_element in movie_elements:
        #extract the title
        title = movie_element.text.strip()
        #extract year
        year = movie_element.find_next("span").text.strip()
        #append title and year to list
        movies.append([title, year])

    # print in terminal what is going to csv
    print("Movie data:\n")
    print("{:<40} {:<10}".format("Title", "Year"))
    print("-" * 50)
    for movie in movies:
        print("{:<40} {:<10}".format(movie[0], movie[1]))

    # write to csv
    filename = "C:/Users/justo/module1dataset/movies_data.csv"
    # open the csv in write mode with utf 8 encoding
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        #Create CSV writer
        writer = csv.writer(csvfile)
        #Write to header row
        writer.writerow(["Title", "Year"])
        # Write the movie data to csv
        writer.writerows(movies)

    print("\nMovie data sent to csv", filename)
        # Extract the movie titles and years for the bar graph
    titles = [movie[0] for movie in movies]
    # Convert the year values to integers if possible, or keep them as strings
    years = [int(movie[1]) if movie[1].isdigit() else movie[1] for movie in movies]

    # Create a bar graph
    plt.bar(titles, years)
    plt.xlabel("Movie Title")
    plt.ylabel("Year")
    plt.title("Top Movies of All Time")
    plt.xticks(rotation=90)
    plt.show()
    
else:
    print("Cannot find data on html. Fix the table you are looking for.")
