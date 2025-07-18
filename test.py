from crud.person import get_all_persons_id_name
from database import SessionLocal

# Create a database session
db = SessionLocal()

# Call the function with the session
persons = get_all_persons_id_name(db)

# Don't forget to close the session
db.close()

print(persons)
