"""
Quick script to set password for a user in the database
Usage: python set_password.py
"""

from database import SessionLocal
from models.person import Person
from services.auth import hash_password

def set_user_password(email: str, password: str):
    """Set password for a user"""
    db = SessionLocal()
    try:
        # Find user by email
        user = db.query(Person).filter(Person.email == email).first()
        
        if not user:
            print(f"❌ User with email '{email}' not found!")
            return False
        
        # Hash the password
        hashed_password = hash_password(password)
        
        # Update user password
        user.password_hash = hashed_password
        db.commit()
        
        print(f"✅ Password set successfully for {user.name} ({email})")
        print(f"   Role: {user.role}")
        print(f"   New password: {password}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    # Set password for alex@gmail.com
    set_user_password("alex@gmail.com", "temp12345")
