# Import the sys module to access system-specific information.
import os
import sys
import pathlib
import re
import subprocess

def checkEndianess():
        # Check if the byte order of the platform is "little" (e.g., Intel, Alpha) and display a corresponding message.
    if sys.byteorder == "little":
        return 1;
        # print("Little-endian platform.")
    else:
        # If the byte order is not "little," assume it's "big" (e.g., Motorola, SPARC) and display a corresponding message.
        # print("Big-endian platform.")
        return 2;
    
    # Display another blank line for clarity.
    print();

def ChkDirAndCreate(dir_path,op):
    if not pathlib.Path(dir_path).is_dir():
        print(f"{dir_path} does not exist, create...");
        if op == 1:
            os.mkdir(dir_path);
        else:
            os.makedirs(dir_path)
        return 1;
    else:
        print(f"{dir_path} has existed...")
        return 2;


def getFileList(srcDir,regex='.*\.wav'):
    # example: regex = '.*\.mp3'
    results = os.listdir(srcDir)
    out_files = []
    cnt_files = 0
    for file in results:
        if os.path.isdir(os.path.join(srcDir, file)):
            out_files += getFileList(os.path.join(srcDir, file))
        elif re.match(regex, file,  re.I):  # file.startswith(startExtension) or file.endswith(".txt") or file.endswith(endExtension):
            out_files.append(os.path.join(srcDir, file))
            cnt_files = cnt_files + 1
    return out_files


def ConvertSR(src_wav, dest_wav, sr):
    subprocess.call('ffmpeg -i {} -ac 1 -ar {} -loglevel error -y {}'.format(
            src_wav, sr, dest_wav), shell=True);

def getFolderList(rootDir=None, recursive=False):
    if not recursive:
        return next(os.walk(rootDir));
    else:
        return [x[0] for x in os.walk(rootDir)]


    