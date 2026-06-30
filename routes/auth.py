from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.person import Person
from schemas.auth import AuthResponse, AuthUser, LoginRequest, SignupRequest
from services.auth import create_access_token, get_current_user, hash_password, verify_password


router = APIRouter(prefix="/auth", tags=["Auth"])


def _normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def _auth_user(user: Person) -> AuthUser:
    return AuthUser(
        id=user.id,
        name=user.name,
        email=user.email,
        phone=user.phone,
        role=user.role,
    )


@router.post("/signup", response_model=AuthResponse)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    email = _normalize_email(payload.email)
    if "@" not in email:
        raise HTTPException(status_code=422, detail="A valid email address is required")

    existing = db.query(Person).filter(Person.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    user = Person(
        name=payload.name.strip(),
        email=email,
        phone=payload.phone,
        phone_number=payload.phone,
        password_hash=hash_password(payload.password),
        role="client",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(subject=str(user.id), role=user.role)
    return AuthResponse(access_token=token, user=_auth_user(user))


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    email = _normalize_email(payload.email)
    user = db.query(Person).filter(Person.email == email).first()
    if not user or not user.password_hash or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is inactive")

    token = create_access_token(subject=str(user.id), role=user.role)
    return AuthResponse(access_token=token, user=_auth_user(user))


@router.get("/me", response_model=AuthUser)
def me(current_user: Person = Depends(get_current_user)):
    return _auth_user(current_user)
