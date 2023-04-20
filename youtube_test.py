# importing the module
from pytube import YouTube, exceptions
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
import time


# where to save
SAVE_PATH = os.getcwd()

def download_video(id):
    # link of the video to be downloaded
    link=f"https://www.youtube.com/watch?v={id}"

    try:
        # object creation using YouTube
        # which was imported in the beginning
        yt = YouTube(link)
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
    total = len(ids)
    for index, id in enumerate(ids):
        link = f"https://www.youtube.com/watch?v={id}"

        try:
            # object creation using YouTube
            # which was imported in the beginning
            yt = YouTube(link)
        except:
            print("Connection Error")  # to handle exception
        print(f"{id}    {index}/{total}")

        success = False
        fail_count = 0
        while success == False and fail_count < 8:
            try:
                mp4files = yt.streams.filter(file_extension='mp4', resolution='360p')
                success = True
            except KeyError:
                print("retrying")
                success = False
                fail_count += 1
            except exceptions.VideoUnavailable:
                print(f"{id} is unavailable")
                unavail_ids.append(id)
                unavail = True
                break

    revised_ids = [x for x in ids if x not in unavail_ids]
    with open('avail.txt', 'w') as f:
        f.write('\n'.join(revised_ids))
        # f.write(str(revised_ids))

    with open('unavail.txt', 'w') as f:
        f.write('\n'.join(unavail_ids))

        # f.write(str(unavail_ids))

    return unavail_ids



        # filters out all the files with "mp4" extension and 360p resolution


# download_video("n2Y3GoN2PGw")
ids = get_ids("ids.txt")
unavail = check_ids(ids)
print(unavail)
