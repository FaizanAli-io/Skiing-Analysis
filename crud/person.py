from sqlalchemy.orm import Session
from models.person import Person
from schemas.person import PersonCreate
from fastapi import HTTPException
def create_person(db: Session, person: PersonCreate):

    existing_person = db.query(Person).filter(Person.email == person.email).first()
    if existing_person:
        raise HTTPException(status_code=400, detail="Email already registered")
    db_person = Person(**person.dict())
    db.add(db_person)
    db.commit()
    db.refresh(db_person)
    return db_person

def get_person(db: Session, person_id: int):
    return db.query(Person).filter(Person.id == person_id).first()

def get_all_persons(db: Session, skip: int = 0, limit: int = 10):
    return db.query(Person).offset(skip).limit(limit).all()

def update_person(db: Session, person_id: int, updated_data: PersonCreate):
    db_person = db.query(Person).filter(Person.id == person_id).first()
    if db_person:
        for key, value in updated_data.dict().items():
            setattr(db_person, key, value)
        db.commit()
        db.refresh(db_person)
    return db_person

def delete_person(db: Session, person_id: int):
    db_person = db.query(Person).filter(Person.id == person_id).first()
    if db_person:
        db.delete(db_person)
        db.commit()
    return db_person
