import os
from pathlib import Path
from multiprocessing import Process, Queue, set_start_method
from dac import DAC
from dac.utils import download
from audiotools import AudioSignal
from tqdm import tqdm

def process_file(queue, gpu_id, model_type="16khz", log_file="skipped_files.log"):
    """
    Worker function to process files using a specific GPU.

    Args:
        queue (Queue): Queue containing file paths to process.
        gpu_id (int): GPU ID to use for processing.
        model_type (str): DAC model type (default is "16khz").
        log_file (str): Path to the log file for skipped files.
    """
    device = f"cuda:{gpu_id}"
    model_path = download(model_type=model_type)
    model = DAC.load(model_path)
    model.to(device)

    while not queue.empty():
        try:
            input_file_path, output_file_path = queue.get_nowait()

            # Skip processing if the output file already exists
            if os.path.exists(output_file_path):
                print(f"Skipping already processed file: {output_file_path}")
                continue

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            # Process the audio file
            signal = AudioSignal(input_file_path)
            signal.to(model.device)

            # Encode and decode the audio signal
            x = model.preprocess(signal.audio_data, signal.sample_rate)
            z, codes, latents, _, _ = model.encode(x, n_quantizers=1)
            decoded_audio = model.decode(z)

            # Save the decoded audio to the output directory as .flac
            decoded_signal = AudioSignal(decoded_audio.cpu().detach().numpy(), signal.sample_rate)
            decoded_signal.write(output_file_path)  # Ensure output_file_path ends with .flac

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"Skipping file due to CUDA OOM: {input_file_path}")
                with open(log_file, "a") as log:
                    log.write(f"CUDA OOM: {input_file_path}\n")
            else:
                print(f"Error processing {input_file_path}: {e}")
        except Exception as e:
            print(f"Error processing {input_file_path}: {e}")
            with open(log_file, "a") as log:
                log.write(f"Error: {input_file_path} - {e}\n")

def process_audio_files_parallel(input_dir, output_dir, num_gpus, model_type="16khz", log_file="skipped_files.log"):
    """
    Parallelizes the processing of audio files across multiple GPUs.

    Args:
        input_dir (str): Path to the input directory containing audio files.
        output_dir (str): Path to the output directory to save processed files.
        num_gpus (int): Number of GPUs to use for parallel processing.
        model_type (str): DAC model type (default is "16khz").
        log_file (str): Path to the log file for skipped files.
    """
    # Create a queue to hold file paths
    queue = Queue()
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".flac"):  # Only process .flac files
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_file_path = os.path.join(output_dir, relative_path, file)

                # Skip adding to the queue if the output file already exists
                if os.path.exists(output_file_path):
                    print(f"Skipping already processed file: {output_file_path}")
                    continue

                queue.put((input_file_path, output_file_path))

    # Create and start processes for each GPU
    processes = []
    for gpu_id in range(num_gpus):
        p = Process(target=process_file, args=(queue, gpu_id, model_type, log_file))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn'
    set_start_method("spawn", force=True)

    num_gpus = 6  # Adjust this based on the number of available GPUs

    input_dir = "/data/anakuzne/LibriSpeech/train-clean-100"
    output_dir = "/data/anakuzne/datasets/LS250BPS/train-clean-100"
    process_audio_files_parallel(input_dir, output_dir, num_gpus, log_file="train_clean_100_skipped.log")

    input_dir = "/data/anakuzne/LibriSpeech/train-clean-360"
    output_dir = "/data/anakuzne/datasets/LS250BPS/train-clean-360"
    process_audio_files_parallel(input_dir, output_dir, num_gpus, log_file="train_clean_360_skipped.log")
    
    input_dir = "/data/anakuzne/LibriSpeech/train-other-500"
    output_dir = "/data/anakuzne/datasets/LS250BPS/train-other-500"
    process_audio_files_parallel(input_dir, output_dir, num_gpus, log_file="train_other_500_skipped.log")

    input_dir = "/data/anakuzne/LibriSpeech/dev-clean"
    output_dir = "/data/anakuzne/datasets/LS250BPS/dev-clean"
    process_audio_files_parallel(input_dir, output_dir, num_gpus, log_file="dev_clean_skipped.log")
    
    input_dir = "/data/anakuzne/LibriSpeech/dev-other"
    output_dir = "/data/anakuzne/datasets/LS250BPS/dev-other"
    process_audio_files_parallel(input_dir, output_dir, num_gpus, log_file="dev_other_skipped.log")
    
    input_dir = "/data/anakuzne/LibriSpeech/test-clean"
    output_dir = "/data/anakuzne/datasets/LS250BPS/test-clean"
    process_audio_files_parallel(input_dir, output_dir, num_gpus, log_file="test_clean_skipped.log")
    
    input_dir = "/data/anakuzne/LibriSpeech/test-other"
    output_dir = "/data/anakuzne/datasets/LS250BPS/test-other"
    process_audio_files_parallel(input_dir, output_dir, num_gpus, log_file="test_other_skipped.log")