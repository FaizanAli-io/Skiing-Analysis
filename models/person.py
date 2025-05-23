from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from database import Base

class Person(Base):
    __tablename__ = 'persons'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))
    email = Column(String(100), unique=True, index=True)
    phone = Column(String(20))

    videos = relationship("VideoAnalysis", back_populates="person")
