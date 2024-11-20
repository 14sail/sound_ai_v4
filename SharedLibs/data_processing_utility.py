import wave
import numpy as np
import array

def save_wavefile(output_path, sig, sr):
    output_file = wave.Wave_write(output_path)
    output_file.setnchannels(1);
    output_file.setframerate(sr);
    output_file.setsampwidth(2) #2bytes per sample.
    # output_file.setparams(params) #nchannels, sampwidth, framerate, nframes, comptype, compname
    output_file.writeframes(array.array('h', sig.astype(np.int16)).tobytes(), )
    output_file.close()

# samplerate = 44100
#The writeWav need to rewrite
# def writeWav():
#     # Put the channels together with shape (2, 44100).
#     audio = np.array([left_channel, right_channel]).T
#     # Convert to (little-endian) 16 bit integers.
#     audio = (audio * (2 ** 15 - 1)).astype("<h")
#     with wave.open("sound1.wav", "w") as f:
#         # 2 Channels.
#         f.setnchannels(2)
#         # 2 bytes per sample.
#         f.setsampwidth(2)
#         f.setframerate(samplerate)
#         f.writeframes(audio.tobytes())


