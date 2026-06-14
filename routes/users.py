from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import crud.person as person_crud
import crud.video_analysis as video_crud
import schemas.person as person_schemas
import schemas.video_analysis as video_schemas
from database import get_db

router = APIRouter(prefix="/users", tags=["Users"])


@router.post("/", response_model=person_schemas.PersonOut)
def create_user(user: person_schemas.PersonCreate, db: Session = Depends(get_db)):
    return person_crud.create_person(db, user)


@router.get("/", response_model=List[person_schemas.PersonOut])
def list_users(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    return person_crud.get_all_persons(db, skip, limit)


@router.get("/{user_id}", response_model=person_schemas.PersonOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = person_crud.get_person(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/{user_id}/attempts", response_model=List[video_schemas.VideoAnalysisOut])
def get_user_attempts(user_id: int, skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    user = person_crud.get_person(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return video_crud.get_person_attempts(db, user_id, skip, limit)
