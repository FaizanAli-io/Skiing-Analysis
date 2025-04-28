from sqlalchemy.orm import Session
from models.video_analysis import VideoAnalysis
from schemas.video_analysis import VideoAnalysisCreate

def create_video_analysis(db: Session, video: VideoAnalysisCreate):
    db_video = VideoAnalysis(**video.dict())
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
    return db_video

def get_video(db: Session, video_id: int):
    return db.query(VideoAnalysis).filter(VideoAnalysis.id == video_id).first()

def get_all_videos(db: Session, skip: int = 0, limit: int = 10):
    return db.query(VideoAnalysis).offset(skip).limit(limit).all()

def update_video(db: Session, video_id: int, updated_data: VideoAnalysisCreate):
    db_video = db.query(VideoAnalysis).filter(VideoAnalysis.id == video_id).first()
    if db_video:
        for key, value in updated_data.dict().items():
            setattr(db_video, key, value)
        db.commit()
        db.refresh(db_video)
    return db_video

def delete_video(db: Session, video_id: int):
    db_video = db.query(VideoAnalysis).filter(VideoAnalysis.id == video_id).first()
    if db_video:
        db.delete(db_video)
        db.commit()
    return db_video
