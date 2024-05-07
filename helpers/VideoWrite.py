from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import os

def write_video(image_files: list, fps: float) -> None:
    """
    Writes a video file from a list of image files.

    Args:
        image_files (list): List of image file paths.
        fps (float): Frames per second for the output video.

    Returns:
        None
    """
    
    output_dir = os.path.abspath("output/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    name = os.path.join(output_dir, 'video_enhanced.mp4')

    clip = ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(name, logger='bar')

    print(f"Video saved to: {name}")

if __name__ == '__main__':
    files = [os.path.abspath('enhance/'+x) for x in os.listdir(os.path.abspath('enhance/'))]
    write_video(files, 23.98*2)