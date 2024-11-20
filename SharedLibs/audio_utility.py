import wave
import pydub
import array
import numpy as np
import wavio


# def wavio_read_wav(src_wav=None):
# 	tmp_data = wavio.read(src_wav).data.T[0];
#     len_data = len(tmp_data);
#     return tmp_data, tmp_data

def save_wavefile(output_path=None, sig=None, sr=20000, channelNum=1, sampleWidth=2):
	with wave.open(output_path,'wb') as f:
	    # output_file = wave.Wave_write(output_path)
	    # output_file.setnchannels(1);
	    f.setnchannels(channelNum);
	    # output_file.setframerate(20000);
	    f.setframerate(sr);
	    # output_file.setsampwidth(2) #2bytes per sample.
	    f.setsampwidth(sampleWidth);
	    # output_file.setparams(params) #nchannels, sampwidth, framerate, nframes, comptype, compname
	    f.writeframes(array.array('h', sig.astype(np.int16)).tobytes(), )
	    # output_file.close();

def read_wavfile_by_wavio(wavfile=None):
    wobj = wavio.read(wavfile);
    sig = wobj.data.T[0];
    return wobj, sig 

