import argparse
import os
from typing import Optional

from pytube import YouTube


def main(url: str, output_path: Optional[str], file_name: Optional[str]) -> None:
    yt = YouTube(url)
    stream = yt.streams.get_highest_resolution()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    stream.download(output_path=output_path, filename=file_name)
    final_name = file_name if file_name else yt.title
    final_path = output_path if output_path else "current directory"
    print(f"Download completed! Video saved as '{final_name}' in '{final_path}'.")