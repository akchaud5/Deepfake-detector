import cv2
import os
from retinaface import RetinaFace
import glob

def extract_faces(frame, model):
    # Detect faces in the frame
    faces = RetinaFace.detect_faces(frame, model=model)
    face_images = []
    if isinstance(faces, dict):
        for key, value in faces.items():
            facial_area = value['facial_area']
            face_img = frame[facial_area[1]:facial_area[3], facial_area[0]:facial_area[2]]
            face_images.append(face_img)
    return face_images

def process_video(video_path, output_dir, video_name):
    # Load the RetinaFace model
    model = RetinaFace.build_model()

    # Open a video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    face_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            face_images = extract_faces(frame, model)
            for face_img in face_images:
                face_filename = os.path.join(output_dir, f'{video_name}-{face_count:02d}.jpg')
                cv2.imwrite(face_filename, face_img)
                face_count += 1
        else:
            break

    # Release everything when job is finished
    cap.release()
    cv2.destroyAllWindows()
    print(f'Faces from video {video_path} saved as {face_filename}')

def check_processed(output_dir, video_name):
    # Check if any faces have been extracted and saved for this video
    search_pattern = os.path.join(output_dir, f'{video_name}-*.jpg')
    return len(glob.glob(search_pattern)) > 0

def process_directory(dataset_dir, output_base_dir):
    # Walk through the directory structure
    for root, dirs, files in os.walk(dataset_dir):
        video_files = [f for f in files if f.endswith('.mp4')]
        video_files.sort()  # Optional: sort files if needed
        video_files = video_files[:20]  # Limit to process only the first 20 videos per folder

        for video_file in video_files:
            video_path = os.path.join(root, video_file)
            relative_path = os.path.relpath(root, dataset_dir)
            output_dir = os.path.join(output_base_dir, relative_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            video_name = os.path.splitext(video_file)[0]
            if not check_processed(output_dir, video_name):
                process_video(video_path, output_dir, video_name)

# Example usage
dataset_dir = '/content/drive/MyDrive/Celeb-DF-v2'
output_base_dir = '/content/drive/MyDrive/RECCE-main/celebdb'
process_directory(dataset_dir, output_base_dir)
