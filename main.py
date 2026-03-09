from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import threading
import uuid
import os

app = Flask(__name__)
CORS(app)
jobs = {}

def process_video(job_id, video_path):
    try:
        jobs[job_id] = {'status': 'processing', 'progress': 0}
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_path = f'/tmp/{job_id}_output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        studio_bg = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            val = int(30 + (i / height) * 50)
            studio_bg[i, :] = [val + 40, val + 20, val]
        backSub = cv2.createBackgroundSubtractorMOG2(
            history=50, varThreshold=40, detectShadows=False
        )
        temp_cap = cv2.VideoCapture(video_path)
        for _ in range(min(30, int(total_frames))):
            ret, frame = temp_cap.read()
            if not ret:
                break
            backSub.apply(frame)
        temp_cap.release()
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            fg_mask = backSub.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.GaussianBlur(fg_mask, (7, 7), 0)
            mask_3ch = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR) / 255.0
            output_frame = (frame * mask_3ch + studio_bg * (1 - mask_3ch)).astype(np.uint8)
            out.write(output_frame)
            frame_count += 1
            if total_frames > 0:
                jobs[job_id]['progress'] = int((frame_count / total_frames) * 100)
        cap.release()
        out.release()
        jobs[job_id] = {'status': 'done', 'progress': 100, 'output': output_path}
    except Exception as e:
        jobs[job_id] = {'status': 'error', 'message': str(e)}

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['video']
    job_id = str(uuid.uuid4())
    video_path = f'/tmp/{job_id}_input.mp4'
    file.save(video_path)
    thread = threading.Thread(target=process_video, args=(job_id, video_path))
    thread.start()
    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>')
def status(job_id):
    return jsonify(jobs.get(job_id, {'status': 'not_found'}))

@app.route('/download/<job_id>')
def download(job_id):
    job = jobs.get(job_id)
    if job and job['status'] == 'done':
        return send_file(job['output'], as_attachment=True, download_name='processed_video.mp4')
    return jsonify({'error': 'Not ready'}), 404

@app.route('/')
def home():
    return jsonify({'status': 'ContentAI server is running!'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
