# importing the module
from pytube import YouTube, exceptions
import os
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
import time
import numpy as np



# where to save
SAVE_PATH = os.getcwd()
referrer = 'https://www.youtube.com/'

# Threshold for mean pixel value below which the frame is considered black
BLACK_FRAME_THRESHOLD = 10

resolution = 360


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


# download_video("n2Y3GoN2PGw")
# ids = get_ids("ids.txt")
# unavail = check_ids(ids)
# print(unavail)

def get_dataset_ids(data_path):
    ids = []
    with open(data_path, encoding="ISO-8859-1") as file:
        for line in file:
            line = eval(line)
            ids.append(line[3])
    return ids
def download_frames_retry(folder_path, audio_folder, video_ids, num_frames):
    if os.path.exists(folder_path) == False or os.path.exists(audio_folder) == False:
        print("no")
        return False

    num_frames+=1
    # Path to the folder where the frames will be saved
    frames_folder = folder_path
    MAX_ATTEMPTS = 5
    failed_ids = []
    total = len(video_ids)
    for index, id in enumerate(video_ids):
        frame_max = num_frames

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
                stream = yt.streams.filter(res=f"{resolution}p").first()
                audio_stream = yt.streams.filter(only_audio=True).first()
                print(yt.streams.filter(only_audio=True))


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

                while True:
                    # Set the frame position in the video
                    frame_pos = frame_count * skip_frames
                    skip_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / num_frames)
                    # Set the video capture object to the frame position
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

                    # Read the frame from the video
                    ret, frame = cap.read()

                    # If the frame is all black, skip it
                    # Convert the frame to grayscale and calculate its mean value
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mean_value = np.mean(gray_frame)

                    # If the mean value is less than 10, skip the frame
                    if mean_value < 0.5:
                        print("black")
                        frame_count += 1
                        frame_max += 1

                        continue

                    # Save the frame to the frames folder
                    frame_file_path = os.path.join(frames_video_folder, f"{id}_{frame_num}.jpg")
                    frame_num += 1
                    cv2.imwrite(frame_file_path, frame)

                    # Increment the frame count
                    frame_count += 1

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

            except ConnectionResetError:
                print("Connection error, retrying")
                success = False
                fail_count += 1
                if fail_count == 15:
                    print(f"connection couldn't get {id}")
                    failed_ids.append([id, 1])

            except exceptions.VideoUnavailable:
                print(f"{id} is unavailable")
                failed_ids.append([id, 2])
                break
    # Loop through each video ID
    return failed_ids

def download_frames(folder_path, video_ids, num_frames):
    num_frames+=1
    # Path to the folder where the frames will be saved
    frames_folder = folder_path
    MAX_ATTEMPTS = 5

    # Loop through each video ID
    for video_id in video_ids:
        frame_max = num_frames

        try:
            # Create a YouTube object
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")

            # Get the video stream with the lowest resolution and its URL
            stream = yt.streams.filter(res="360p").first()
            video_url = stream.url

            # Create a VideoCapture object from the video URL
            cap = cv2.VideoCapture(video_url)

            # Create the frames folder for this video ID
            frames_video_folder = os.path.join(frames_folder, video_id)
            os.makedirs(frames_video_folder, exist_ok=True)

            # Get the frames from the video and save them to the frames folder
            frame_count = 0
            frame_num = 0
            skip_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / num_frames)

            while True:
                # Set the frame position in the video
                frame_pos = frame_count * skip_frames
                skip_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / num_frames)
                # Set the video capture object to the frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

                # Read the frame from the video
                ret, frame = cap.read()

                # If the frame is all black, skip it
                # Convert the frame to grayscale and calculate its mean value
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_value = np.mean(gray_frame)

                # If the mean value is less than 10, skip the frame
                if mean_value < 0.5:
                    print("black")
                    frame_count += 1
                    frame_max +=1

                    continue

                # Save the frame to the frames folder
                frame_file_path = os.path.join(frames_video_folder, f"{video_id}_{frame_num}.jpg")
                frame_num+=1
                cv2.imwrite(frame_file_path, frame)

                # Increment the frame count
                frame_count += 1

                # Break the loop if we have saved 5 frames
                if frame_count == num_frames:
                    break

            # Release the video capture object
            cap.release()

            print(f"Downloaded frames for video ID {video_id}")
        except Exception as e:
            print(e)
            print(f"Failed to download frames for video ID {video_id}")
def main():

    ids = get_dataset_ids("dataset.txt")
    # print(ids)
    test = ids[3:5]
    print(test)
    download_frames_retry("frames/", "audio/", test,8)


main()


