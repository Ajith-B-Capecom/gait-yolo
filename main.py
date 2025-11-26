from fastapi import FastAPI
from fastapi.responses import JSONResponse 
import os
from scripts.yolopose_seg import process_videos_with_pose_detection
app = FastAPI()

VIDEOS_FOLDER = "videos"
OUTPUT_FOLDER = "data"

POSE_MODEL_PATH = "yolo11n-pose.pt"
SEG_MODEL_PATH = "yolo11n-seg.pt"

SKIP_FRAMES = 3
CONF_THRESHOLD = 0.5
LINE_THICKNESS = 3
EXTRACTION_MODE = "both"
APPLY_MORPHOLOGY = True
TRACKER_TYPE = 'botsort.yaml'

@app.get("/")
async def run_processing():
    try:
        process_videos_with_pose_detection(
            videos_folder=VIDEOS_FOLDER,
            model_path=POSE_MODEL_PATH,
            seg_model_path=SEG_MODEL_PATH,
            skip_frames=SKIP_FRAMES,
            conf_threshold=CONF_THRESHOLD,
            line_thickness=LINE_THICKNESS,
            extraction_mode=EXTRACTION_MODE,
            output_folder=OUTPUT_FOLDER,
            apply_morphology=APPLY_MORPHOLOGY,
            tracker_type=TRACKER_TYPE
        )

        return JSONResponse({"status": "success", "message": "Processing completed."})

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
if __name__ == "__main__":
    main()