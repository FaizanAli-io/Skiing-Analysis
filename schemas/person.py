from pydantic import BaseModel
from typing import Optional

class PersonCreate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None

class PersonOut(PersonCreate):
    id: int
    role: Optional[str] = "client"
    is_active: Optional[bool] = True

    class Config:
        from_attributes = True
        


class PersonIDName(BaseModel):
    id: int
    name: str
