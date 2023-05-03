import requests
from pytube import YouTube, exceptions
import os
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
import time
import numpy as np

api_key = "779ffcbce4723a30bcec338f0ff0ba18"
max_page_number = 80
page_size = 50

# where to save
SAVE_PATH = os.getcwd()
referrer = 'https://www.youtube.com/'

# Threshold for mean pixel value below which the frame is considered black
BLACK_FRAME_THRESHOLD = 10

resolution = 360

def collect_data(page_max):
    with open("dataset.txt", "a") as file:
        # url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={query}"
        # page_url = f"https://api.themoviedb.org/3/movie/popular?api_key={api_key}&page={page_number}&page_size={page_size}
        # details = requests.get(f"https://api.themoviedb.org/3/movie/{key}?api_key={api_key}&append_to_response=videos")

        movie_data_list = []

        for page_number in range(1,page_max):
            page_url = f"https://api.themoviedb.org/3/movie/popular?api_key={api_key}&page={page_number}&page_size={page_size}&language=en"
            page_response = requests.get(page_url)

            if page_response.status_code == 200:
                page_data = page_response.json()
                for movie in page_data["results"]:
                    if movie["original_language"] == "en":
                        title = movie["original_title"]
                        key = movie["id"]
                        conv_genres = process_genres(movie["genre_ids"])
                        if no_genres(conv_genres) == True:
                            continue
                        movie_data = [title, movie["id"], conv_genres]
                        details = requests.get(f"https://api.themoviedb.org/3/movie/{key}?api_key={api_key}&append_to_response=videos")
                        details_data = details.json()

                        youtube_ids = details_data["videos"]["results"]
                        vid = None
                        for youtube_id in youtube_ids:
                            if youtube_id["type"] == "Trailer" and youtube_id["official"]==True:
                                vid = youtube_id
                                break

                        if vid:
                            movie_data.append(vid["key"])
                            print(movie_data, convert_genres(movie_data[2]))
                            try:
                                file.write(str(movie_data) + "\n")
                            except UnicodeEncodeError:
                                continue

                            movie_data_list.append(movie_data)

            else:
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

    return output

def convert_genres(genres):
    genre_list = []
    genre_names = ['Action', 'Adventure', 'Comedy', 'Crime', 'Drama', 'Horror', 'Mystery', 'Romance', 'Science Fiction', 'Thriller']

    for i in range(len(genres)):
        if genres[i] == 1:
            genre_list.append(genre_names[i])

    return genre_list

def no_genres(arr):
    for element in arr:
        if element != 0:
            return False
    return True

def download_video(id):
    # link of the video to be downloaded
    link=f"https://www.youtube.com/watch?v={id}"

    try:
        # object creation using YouTube
        # which was imported in the beginning
        yt = YouTube(link, referrer=referrer)
    except:
        print("Connection Error") #to handle exception

    # filters out all the files with "mp4" extension and 360p resolution


    success = False
    fail_count = 0
    while success == False and fail_count < 8:
        try:
            mp4files = yt.streams.filter(file_extension='mp4', resolution='360p')
            success=True
        except KeyError:
            print("failed, retrying")
            success=False
            fail_count+=1
    #
    if success == True:
        title = mp4files.first().title
        print(f"downloading {title}")
        # download the first stream with mp4 and 360p resolution
        mp4files.first().download(output_path=SAVE_PATH, filename=f"{id}.mp4")
        print(f'{title} added to {SAVE_PATH}')
        return True
    else:
        print(f"{id} failed after {fail_count} tries")
        return False

def separate_audio(video_path, audio_path):
    # create a VideoFileClip object
    video_clip = VideoFileClip(video_path)

    # extract audio from the video
    audio_clip = video_clip.audio

    # write the audio to a file
    audio_clip.write_audiofile(audio_path)

    # close the clips
    video_clip.close()
    audio_clip.close()
    if os.path.exists(audio_path):
        return True
    return False

def get_ids(source):
    with open(source, 'r') as file:
        lines = file.readlines()
    list_items = [line.strip() for line in lines]
    return list_items

def check_ids(ids):
    unavail_ids = []
    unlisted = []
    total = len(ids)
    with open('avail.txt', 'a') as avail_file, open('unavail.txt', 'a') as unavail_file:

        for index, id in enumerate(ids):
            link = f"https://www.youtube.com/watch?v={id}"

            try:
                # object creation using YouTube
                # which was imported in the beginning
                yt = YouTube(link)
            except:
                print("Connection Error")  # to handle exception
            print(f"{id}    {index+1}/{total}")

            success = False
            fail_count = 0
            while success == False and fail_count < 15:
                try:
                    mp4files = yt.streams.filter(file_extension='mp4', resolution='360p')
                    success = True
                    avail_file.write(str(id) + '\n')
                except KeyError:
                    print("retrying")
                    success = False
                    fail_count += 1
                    if fail_count == 15:
                        print(f"can't access {id}")
                        unlisted.append(id)
                        unavail_file.write(str(id) + '\n')

                except ConnectionResetError:
                    print("Connection error, retrying")
                    success = False
                    fail_count += 1
                    if fail_count == 15:
                        print(f"connection coudln't get {id}")
                        unlisted.append(id)
                        unavail_file.write(str(id) + '\n')


                except exceptions.VideoUnavailable:
                    print(f"{id} is unavailable")
                    unavail_ids.append(id)
                    unavail_file.write(str(id) + '\n')
                    unavail = True
                    break

    # revised_ids = [x for x in ids if x not in unavail_ids and x not in unlisted]
    # with open('avail.txt', 'w') as f:
    #     f.write('\n'.join(revised_ids))
    #     # f.write(str(revised_ids))
    #
    # with open('unavail.txt', 'w') as f:
    #     f.write('\n'.join(unavail_ids))
    #     f.write("\n unlisted \n")
    #     f.write('\n'.join(unlisted))


        # f.write(str(unavail_ids))



    return unavail_ids



        # filters out all the files with "mp4" extension and 360p resolution

def get_dataset_ids(data_path):
    ids = []
    with open(data_path, encoding="ISO-8859-1") as file:
        for line in file:
            line = eval(line)
            ids.append(line[3])
    return ids

def get_failed(data_path):
    ids = []
    with open(data_path, encoding="ISO-8859-1") as file:
        for line in file:
            line = eval(line)
            ids.append(line[0])
    return ids

def download_frames_retry(folder_path, audio_folder, video_ids, num_frames):
    if os.path.exists(folder_path) == False or os.path.exists(audio_folder) == False:
        print("destination paths do not exist")
        print(os.getcwd())
        return False
    start_time = time.time()
    num_frames+=1
    # Path to the folder where the frames will be saved
    frames_folder = folder_path
    MAX_ATTEMPTS = 5
    failed_ids = []
    total = len(video_ids)
    for index, id in enumerate(video_ids):

        success_rate = (1-(len(failed_ids)/(index+1)))*100
        frame_max = num_frames

        link = f"https://www.youtube.com/watch?v={id}"
        try:
            # object creation using YouTube
            # which was imported in the beginning
            yt = YouTube(link)
        except:
            print("Connection Error")  # to handle exception

        end_time = time.time()

        curr_time = (end_time - start_time)/60

        print(f"{id}\t{index+1}/{total}\tSuccessful: {success_rate:.2f}\tElapsed: {curr_time:.2f}m")

        success = False
        fail_count = 0
        with open('unavailable_videos.txt', 'a') as file:
            while success == False and fail_count < 15:
                try:
                    stream = yt.streams.filter(res=f"{resolution}p").first()
                    audio_stream = yt.streams.filter(only_audio=True).first()
                    video_url = stream.url
                    # Download the audio stream to the audio folder
                    # audio_file_path = os.path.join(audio_folder, f"{id}.wav")
                    audio_stream.download(output_path=audio_folder, filename=f"{id}.wav")

                    # Create a VideoCapture object from the video URL
                    cap = cv2.VideoCapture(video_url)

                    # Create the frames folder for this video ID
                    frames_video_folder = os.path.join(frames_folder, id)
                    os.makedirs(frames_video_folder, exist_ok=True)

                    # Get the frames from the video and save them to the frames folder
                    frame_count = 0
                    frame_num = 0
                    skip_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / num_frames)
                    offset = 0

                    while True:

                        # Set the frame position in the video
                        # print(frame_count)

                        frame_pos = frame_count * skip_frames + offset
                        if frame_count == 0:
                            frame_pos += 200
                        # print(frame_pos)
                        # skip_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / num_frames)
                        # Set the video capture object to the frame position
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

                        # Read the frame from the video
                        ret, frame = cap.read()

                        # If the frame is all black, skip it
                        # Convert the frame to grayscale and calculate its mean value
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        mean_value = np.mean(gray_frame)

                        # If the mean value is less than 10, skip the frame
                        # print("max", cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if mean_value < 3 and (frame_pos+offset) < cap.get(cv2.CAP_PROP_FRAME_COUNT):
                            # print("black")
                            offset += 1
                            continue

                        # If the mean value is not less than 0.5, increment the frame count and reset the offset
                        frame_count += 1
                        offset = 0

                        # Save the frame to the frames folder
                        frame_file_path = os.path.join(frames_video_folder, f"{id}_{frame_num}.png")
                        frame_num += 1
                        cv2.imwrite(frame_file_path, frame)

                        # Increment the frame count

                        # Break the loop if we have saved 5 frames
                        if frame_count == num_frames:
                            break

                    # Release the video capture object
                    cap.release()

                    print(f"Downloaded frames for video ID {id}")
                    success = True

                except KeyError:
                    print("retrying")
                    success = False
                    fail_count += 1
                    if fail_count == 15:
                        print(f"can't access {id}")
                        failed_ids.append([id, 0])
                        file.write(f"\n{[id, 0]}")

                except ConnectionResetError:
                    print("Connection error, retrying")
                    success = False
                    fail_count += 1
                    if fail_count == 15:
                        print(f"connection couldn't get {id}")
                        failed_ids.append([id, 1])
                        file.write(f"\n{[id, 1]}")

                except exceptions.VideoUnavailable:
                    print(f"{id} is unavailable")
                    failed_ids.append([id, 2])
                    file.write(f"\n{[id, 3]}")
                    break
                except Exception as e:
                    print(e)

    now_time = time.time()
    complete_time = (now_time - start_time)/60

    succ_num = len(video_ids) - len(failed_ids)

    print(f"Downloaded {succ_num}/{len(video_ids)} in {complete_time:.2f}m")

    # Loop through each video ID
    return failed_ids

def folder_count(folder_path):
    folders = next(os.walk(folder_path))[1]
    num_folders = len(folders)
    print(f"There are {num_folders} folders in {folder_path}.")
    return folders

def file_count(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".wav")]
    num_files = len(files)
    print(f"There are {num_files} files in {folder_path}.")
    return files

def find_undownloaded_files():
    ids = get_dataset_ids("dataset.txt")

    train_folders = folder_count("/mnt/h/My Drive/dl4m_datasets/trailer_dataset/train/frames/")
    test_folders = folder_count("/mnt/h/My Drive/dl4m_datasets/trailer_dataset/test/frames/")
    valid_folders = folder_count("/mnt/h/My Drive/dl4m_datasets/trailer_dataset/validation/frames/")

    fail_ids = get_failed("unavailable_videos.txt")
    collected = len(train_folders)+len(test_folders)+len(valid_folders)+len(fail_ids)

    ids = set(ids)
    coll_ids = set(train_folders + test_folders + valid_folders + fail_ids)

    print(len(train_folders)+len(test_folders)+len(valid_folders)+len(fail_ids))
    print(len(ids) - collected)
    print(len(ids - coll_ids))
    missing_ids = ids - coll_ids
    print(missing_ids)
    return list(missing_ids)

def check_files(frames_path, audio_path):
    folders = folder_count(frames_path)
    files = file_count(audio_path)
    audio_list = [file_name[:-4] for file_name in files]

    # Print the new list
    diff = set(folders) ^ set(audio_list)
    print(len(diff))
    if len(diff) > 0:
        print("The differing item is:", diff.pop())
        return False
    else:
        print("The two lists are identical.")
        return folders

def confirm_dataset():
    check_train = check_files("/mnt/h/My Drive/dl4m_datasets/trailer_dataset/train/frames/",
                        "/mnt/h/My Drive/dl4m_datasets/trailer_dataset/train/audio/"
                        )
    check_test = check_files("/mnt/h/My Drive/dl4m_datasets/trailer_dataset/test/frames/",
                        "/mnt/h/My Drive/dl4m_datasets/trailer_dataset/test/audio/"
                        )

    check_validation = check_files("/mnt/h/My Drive/dl4m_datasets/trailer_dataset/validation/frames/",
                        "/mnt/h/My Drive/dl4m_datasets/trailer_dataset/validation/audio/"
                        )
    if check_train != False and check_test != False and check_validation!=False:
        train = set(check_train)
        test = set(check_test)
        validation = set(check_validation)
        if (
                len(train.intersection(test)) == 0
            and len(train.intersection(validation)) == 0
            and len(test.intersection(validation)) == 0
        ):

            print("No overlapping data")
            return True
        else:
            print("Overlapping data found")
            return train.intersection(test), train.intersection(validation), test.intersection(validation)


    else:
        print("Mismatched frames/audio")
        print(check_train, check_test, check_validation)
        return False


