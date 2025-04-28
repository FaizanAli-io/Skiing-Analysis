from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
import crud.person as crud
import schemas.person as schemas

router = APIRouter(prefix="/persons", tags=["Persons"])

@router.post("/", response_model=schemas.PersonOut)
def create(person: schemas.PersonCreate, db: Session = Depends(get_db)):
    return crud.create_person(db, person)

@router.get("/{person_id}", response_model=schemas.PersonOut)
def read(person_id: int, db: Session = Depends(get_db)):
    db_person = crud.get_person(db, person_id)
    if db_person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return db_person

@router.get("/", response_model=list[schemas.PersonOut])
def read_all(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    return crud.get_all_persons(db, skip, limit)

@router.put("/{person_id}", response_model=schemas.PersonOut)
def update(person_id: int, person: schemas.PersonCreate, db: Session = Depends(get_db)):
    updated = crud.update_person(db, person_id, person)
    if not updated:
        raise HTTPException(status_code=404, detail="Person not found")
    return updated

@router.delete("/{person_id}")
def delete(person_id: int, db: Session = Depends(get_db)):
    deleted = crud.delete_person(db, person_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Person not found")
    return {"message": "Deleted successfully"}
