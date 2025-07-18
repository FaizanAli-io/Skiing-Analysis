from pydantic import BaseModel

class PersonCreate(BaseModel):
    name: str
    email: str
    phone: str

class PersonOut(PersonCreate):
    id: int

    class Config:
        from_attributes = True
        


class PersonIDName(BaseModel):
    id: int
    name: str
