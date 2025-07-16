import os
import argparse
from pathlib import Path
from multiprocessing import Process, Queue, set_start_method
from dac import DAC
from dac.utils import download
from audiotools import AudioSignal
from tqdm import tqdm

def process_file(queue, gpu_id, model_type="16khz", log_file="skipped_files.log", num_quantizers=1):
    """
    Worker function to process files using a specific GPU.

    Args:
        queue (Queue): Queue containing file paths to process.
        gpu_id (int): GPU ID to use for processing.
        model_type (str): DAC model type (default is "16khz").
        log_file (str): Path to the log file for skipped files.
        num_quantizers (int): Number of quantizers to use for encoding.
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
            # Convert to mono if necessary
            if signal.num_channels > 1:
                print(f"Converting {input_file_path} to mono (1 channel)")
                signal = signal.to_mono()

            # Resample to 16kHz if necessary
            if signal.sample_rate != 16000:
                print(f"Resampling {input_file_path} from {signal.sample_rate}Hz to 16kHz")
                signal.resample(16000)

            signal.to(model.device)

            # Encode and decode the audio signal
            x = model.preprocess(signal.audio_data, signal.sample_rate)
            z, codes, latents, _, _ = model.encode(x, n_quantizers=num_quantizers)
            decoded_audio = model.decode(z)

            # Save the decoded audio to the output directory
            decoded_signal = AudioSignal(decoded_audio.cpu().detach().numpy(), signal.sample_rate)
            decoded_signal.write(output_file_path)

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

def process_audio_files_parallel(input_dir, output_dir, num_gpus, model_type="16khz", log_file="skipped_files.log", num_quantizers=1):
    """
    Parallelizes the processing of audio files across multiple GPUs.

    Args:
        input_dir (str): Path to the input directory containing audio files.
        output_dir (str): Path to the output directory to save processed files.
        num_gpus (int): Number of GPUs to use for parallel processing.
        model_type (str): DAC model type (default is "16khz").
        log_file (str): Path to the log file for skipped files.
        num_quantizers (int): Number of quantizers to use for encoding.
    """
    # Create a queue to hold file paths
    queue = Queue()
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".flac") or file.endswith(".wav"):  # Process both .flac and .wav files
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
        p = Process(target=process_file, args=(queue, gpu_id, model_type, log_file, num_quantizers))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawnd'

    set_start_method("spawn", force=True)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process audio files with DAC.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing audio files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory to save processed files.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for parallel processing.")
    parser.add_argument("--model_type", type=str, default="16khz", help="DAC model type (default: 16khz).")
    parser.add_argument("--log_file", type=str, default="skipped_files.log", help="Path to the log file for skipped files.")
    parser.add_argument("--num_quantizers", type=int, default=1, help="Number of quantizers to use for encoding.")
    args = parser.parse_args()

    # Run the processing
    process_audio_files_parallel(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        model_type=args.model_type,
        log_file=args.log_file,
        num_quantizers=args.num_quantizers,
    )