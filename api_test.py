import requests

api_key = "779ffcbce4723a30bcec338f0ff0ba18"
query = "Jack Reacher"
max_page_number = 5
page_size = 50

def collect_data(page_max):
    # url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={query}"
    # page_url = f"https://api.themoviedb.org/3/movie/popular?api_key={api_key}&page={page_number}&page_size={page_size}"
    movie_data_list = []

    for page_number in range(1,page_max):
        page_url = f"https://api.themoviedb.org/3/movie/popular?api_key={api_key}&page={page_number}&page_size={page_size}&language=en"

        # response = requests.get(url)
        page_response = requests.get(page_url)

        if page_response.status_code == 200:
            page_data = page_response.json()
            # key = page_data["results"][0]["id"]

            for movie in page_data["results"]:
                if movie["original_language"] == "en":
                    title = movie["original_title"]
                    key = movie["id"]
                    conv_genres = process_genres(movie["genre_ids"])
                    movie_data = [title, movie["id"], conv_genres]
                    details = requests.get(f"https://api.themoviedb.org/3/movie/{key}?api_key={api_key}&append_to_response=videos")
                    details_data = details.json()

                    youtube_ids = youtube_id = details_data["videos"]["results"]
                    vid = None
                    for youtube_id in youtube_ids:
                        if youtube_id["type"] == "Trailer" and youtube_id["official"]==True:
                            vid = youtube_id
                            break




                    # youtube_id = details_data["videos"]["results"][0]["key"]

                    if vid:
                        movie_data.append(vid["key"])
                        movie_data_list.append(movie_data)
                        print(movie_data, convert_genres(movie_data[2]))





            # details = requests.get(f"https://api.themoviedb.org/3/movie/{key}?api_key={api_key}&append_to_response=videos")
            # details_data = details.json()
            #
            # print(details_data["videos"]["results"][0]["key"])
        else:
            # something went wrong...
            print(f"Error: {page_response.status_code} - {page_response.text}")

    print(len(movie_data_list))
    print(movie_data_list)


def process_genres(genre_ids):
    output = [0] * 10

    for id in genre_ids:
        if id == 28:
            output[0] = 1
        if id == 12:
            output[1] = 1
        if id == 35:
            output[2] = 1
        if id == 80:
            output[3] = 1
        if id == 18:
            output[4] = 1
        if id == 27:
            output[5] = 1
        if id == 9648:
            output[6] = 1
        if id == 10749:
            output[7] = 1
        if id == 878:
            output[8] = 1
        if id == 53:
            output[9] = 1
    return (output)


def convert_genres(genres):
    genre_list = []
    genre_names = ['Action', 'Adventure', 'Comedy', 'Crime', 'Drama', 'Horror', 'Mystery', 'Romance', 'Science Fiction', 'Thriller']

    for i in range(len(genres)):
        if genres[i] == 1:
            genre_list.append(genre_names[i])

    return genre_list

collect_data(10)