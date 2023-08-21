"""
Downloads and creates manifest files for speech recognition with Mini LibriSpeech.

Authors:
 * Peter Plantinga, 2021
 * Mirco Ravanelli, 2021
"""

import os
import sys
import json
import shutil
import logging
from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
MINILIBRI_TRAIN_URL = "http://www.openslr.org/resources/31/train-clean-5.tar.gz"
MINILIBRI_VALID_URL = "http://www.openslr.org/resources/31/dev-clean-2.tar.gz"
MINILIBRI_TEST_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
SAMPLERATE = 16000



def prepare_librispeech(
    data_folder, parts, save_json_train, save_json_valid, save_json_test
):
    """
    Prepares the json files for the Librispeech dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the Mini Librispeech dataset is stored.

    parts: List[str] which subparts of LS to use. Eg. train-clean-100,
        train-clean-360, train-other-500
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.

    Example
    -------
    >>> data_folder = '/path/to/librispeech'
    >>> prepare_librispeech(data_folder, 
        ['train-clean-360', 'train-clean-100'], 'train.json', 'valid.json', 'test.json')
    """

    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # If the dataset doesn't exist yet, download it
    train_folders = [
        os.path.join(data_folder, part) for part in parts
    ]
    valid_folder = os.path.join(data_folder, "dev-clean")
    test_folder = os.path.join(data_folder, "test-clean")

    # Data should be already downloaded, comment it
    # if not check_folders(train_folder, valid_folder, test_folder):
    #    download_mini_librispeech(data_folder)

    # List files and create manifest from list
    logger.info("Creating %s, %s, and %s", save_json_train, save_json_valid, save_json_test)
    extension = [".flac"]

    # List of flac audio files
    wav_list_train = []
    for folder in train_folders:
        # Get files for the current subset
        curr_files = get_all_files(folder, match_and=extension)
        wav_list_train.extend(curr_files)
    wav_list_valid = get_all_files(valid_folder, match_and=extension)
    wav_list_test = get_all_files(test_folder, match_and=extension)

    # List of transcription file
    extension = [".trans.txt"]
    trans_list = get_all_files(data_folder, match_and=extension)
    trans_dict = get_transcription(trans_list)

    # Create the json files
    create_json(wav_list_train, trans_dict, save_json_train)
    create_json(wav_list_valid, trans_dict, save_json_valid)
    create_json(wav_list_test, trans_dict, save_json_test)


def get_transcription(trans_list):
    """
    Returns a dictionary with the transcription of each sentence in the dataset.

    Arguments
    ---------
    trans_list : list of str
        The list of transcription files.
    """
    # Processing all the transcription files in the list
    trans_dict = {}
    for trans_file in trans_list:
        # Reading the text file
        with open(trans_file, encoding="utf-8") as file:
            for line in file:
                uttid = line.split(" ")[0]
                text = line.rstrip().split(" ")[1:]
                text = " ".join(text)
                trans_dict[uttid] = text

    logger.info("Transcription files read!")
    return trans_dict


def create_json(wav_list, trans_dict, json_file):
    """
    Creates the json file given a list of wav files and their transcriptions.

    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    trans_dict : dict
        Dictionary of sentence ids and word transcriptions.
    json_file : str
        The path of the output json file
    """
    # Processing all the wav files in the list
    json_dict = {}
    for wav_file in wav_list:
        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Manipulate path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        relative_path = os.path.join("{data_root}", *path_parts[-5:])

        # Create entry for this utterance
        json_dict[uttid] = {
            "wav": relative_path,
            "length": duration,
            "words": trans_dict[uttid],
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w", encoding="utf-8") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info("%s successfully created!", json_file)

def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def download_mini_librispeech(destination):
    """Download dataset and unpack it.

    Arguments
    ---------
    destination : str
        Place to put dataset.
    """
    train_archive = os.path.join(destination, "train-clean-5.tar.gz")
    valid_archive = os.path.join(destination, "dev-clean-2.tar.gz")
    test_archive = os.path.join(destination, "test-clean.tar.gz")
    download_file(MINILIBRI_TRAIN_URL, train_archive)
    download_file(MINILIBRI_VALID_URL, valid_archive)
    download_file(MINILIBRI_TEST_URL, test_archive)
    shutil.unpack_archive(train_archive, destination)
    shutil.unpack_archive(valid_archive, destination)
    shutil.unpack_archive(test_archive, destination)


def main(data_folder, data_save_folder):
    """Runs data preparation script on the data_folder.
    """
    # data_folder, parts, save_json_train, save_json_valid, save_json_test
    prepare_librispeech(
        data_folder,
        ["train-clean-100", "train-clean-360", "train-other-500"],
        f"{data_save_folder}/train.json",
        f"{data_folder}/valid.json",
        f"{data_save_folder}/test.json"

    )

if __name__=="__main__":
    main(sys.argv[1], sys.argv[2])
