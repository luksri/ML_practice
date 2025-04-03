"""
This version of the tool is using google text to speech module

"""
import os
from moviepy import *
import re
import gtts

text_only_pat = re.compile(r'[^a-zA-Z0-9,\s]')
output_dir = "./video"
os.makedirs(output_dir, exist_ok=True)

def video_gen(texts,image_file_names, video_name):
    audio = []
    for i, text in enumerate(texts):
        if text.strip():
            audio_file = f'{output_dir}/audio/audio{i}.mp3'
            tts = gtts.gTTS(text=text, lang='en')
            tts.save(audio_file)
            audio.append(AudioFileClip(audio_file))

    # concatenate audio clips
    final_audio = concatenate_audioclips(audio)
    final_audio.write_audiofile(f"{output_dir}/narration.aac", codec='aac')

    # create image clips with duration matching audio
    # image_file_names = list(images.values())
    print(image_file_names)

    image_clips = []
    for i, img in enumerate(image_file_names):
        duration = audio[i].duration if audio[i] else 3
        img_clip = ImageClip(img).with_duration(duration)
        image_clips.append(img_clip)

    ## concatenate image clips into video
    video = concatenate_videoclips(image_clips, method='compose').with_audio(
        concatenate_audioclips([clip for clip in audio if clip]))
    output_video = f'{output_dir}/{video_name}.mp4'
    video.write_videofile(output_video, fps=24, codec='libx264', audio_codec='aac')
    print(f"video saved")
