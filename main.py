import os
import socket
from src.match_processor import MatchProcessor

def main():
    hostname = socket.gethostname()

    if "clab" in hostname:
        input_file_path = "/datasets/tdt4265/other/rbk/3_test_1min_hamkam_from_start/img1"
        ssh_mode = True
    else:
        input_file_path = os.path.expanduser("~/Downloads/3_test_1min_hamkam_from_start/img1")
        ssh_mode = False

    output_file_path = "output"
    weights_path = "rbk_weights/weights/best.pt"

    processor = MatchProcessor(
        input_dir=input_file_path,
        output_dir=output_file_path,
        weights_path=weights_path,
        ssh_mode=ssh_mode,
    )
    processor.run()


if __name__ == "__main__":
    main()