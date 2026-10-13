from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.person import Person
from schemas.auth import (
    AuthConfig,
    AuthResponse,
    AuthUser,
    ChangePasswordRequest,
    ForgotPasswordRequest,
    LoginRequest,
    MessageResponse,
    ResetPasswordRequest,
    SignupRequest,
    SignupVerifyRequest,
)
from services import email_service
from services.auth import create_access_token, get_current_user, hash_password, verify_password
from services.otp import CODE_TTL_MINUTES, issue_code, verify_code


router = APIRouter(prefix="/auth", tags=["Auth"])

GENERIC_RESET_MESSAGE = "If an account exists for that email, a reset code has been sent."
INVALID_CODE_MESSAGE = "That code is invalid or has expired. Please request a new one."


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


def _auth_response(user: Person) -> AuthResponse:
    token = create_access_token(
        subject=str(user.id),
        role=user.role,
        token_version=user.token_version or 0,
    )
    return AuthResponse(access_token=token, user=_auth_user(user))


def _require_valid_email(email: str) -> None:
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=422, detail="A valid email address is required")


def _send_code(email: str, code: str, purpose: str) -> None:
    try:
        email_service.send_code_email(email, code, purpose, CODE_TTL_MINUTES)
    except RuntimeError:
        raise HTTPException(status_code=502, detail="We could not send the email right now. Please try again shortly.")


@router.get("/config", response_model=AuthConfig)
def auth_config():
    """Tell the frontend which email-based flows are switched on."""
    enabled = email_service.is_configured()
    return AuthConfig(email_verification_required=enabled, password_reset_available=enabled)


@router.post("/signup", response_model=AuthResponse)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    """Direct signup, used only until email (SMTP) is configured."""
    if email_service.is_configured():
        raise HTTPException(status_code=400, detail="Please verify your email to create an account.")

    email = _normalize_email(payload.email)
    _require_valid_email(email)

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
    return _auth_response(user)


@router.post("/signup/request-code", response_model=MessageResponse)
def signup_request_code(payload: SignupRequest, db: Session = Depends(get_db)):
    """Step 1 of signup: email a code. The account is created only after verification."""
    if not email_service.is_configured():
        raise HTTPException(status_code=503, detail="Email verification is not available yet.")

    email = _normalize_email(payload.email)
    _require_valid_email(email)
    if db.query(Person).filter(Person.email == email).first():
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    code = issue_code(db, email, "signup", payload={
        "name": payload.name.strip(),
        "phone": payload.phone,
        "password_hash": hash_password(payload.password),
    })
    _send_code(email, code, "signup")
    return MessageResponse(message=f"We sent a 6-digit code to {email}.")


@router.post("/signup/verify", response_model=AuthResponse)
def signup_verify(payload: SignupVerifyRequest, db: Session = Depends(get_db)):
    """Step 2 of signup: check the code and create the account."""
    email = _normalize_email(payload.email)
    record = verify_code(db, email, "signup", payload.code)
    if db.query(Person).filter(Person.email == email).first():
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    details = record.payload or {}
    if not details.get("password_hash"):
        raise HTTPException(status_code=400, detail=INVALID_CODE_MESSAGE)

    user = Person(
        name=details.get("name") or email.split("@")[0],
        email=email,
        phone=details.get("phone"),
        phone_number=details.get("phone"),
        password_hash=details["password_hash"],
        role="client",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _auth_response(user)


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    email = _normalize_email(payload.email)
    user = db.query(Person).filter(Person.email == email).first()
    if not user or not user.password_hash or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is inactive")
    return _auth_response(user)


@router.get("/me", response_model=AuthUser)
def me(current_user: Person = Depends(get_current_user)):
    return _auth_user(current_user)


@router.post("/password/forgot", response_model=MessageResponse)
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Email a reset code. The reply is identical whether or not the account exists."""
    if not email_service.is_configured():
        raise HTTPException(status_code=503, detail="Password reset by email is not available yet. Please contact Bluerun.")

    email = _normalize_email(payload.email)
    user = db.query(Person).filter(Person.email == email).first()
    if user and user.is_active and user.email:
        try:
            code = issue_code(db, email, "password_reset")
        except HTTPException as exc:
            if exc.status_code != 429:
                raise
            # Cooldown: a code was just sent. Answer the same way so the
            # response never reveals whether the account exists.
            return MessageResponse(message=GENERIC_RESET_MESSAGE)
        _send_code(email, code, "password_reset")
    return MessageResponse(message=GENERIC_RESET_MESSAGE)


@router.post("/password/reset", response_model=MessageResponse)
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    email = _normalize_email(payload.email)
    verify_code(db, email, "password_reset", payload.code)
    user = db.query(Person).filter(Person.email == email).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=400, detail=INVALID_CODE_MESSAGE)

    user.password_hash = hash_password(payload.new_password)
    # Signs out every existing session for this account.
    user.token_version = (user.token_version or 0) + 1
    db.commit()
    return MessageResponse(message="Your password has been reset. You can sign in now.")


@router.post("/password/change", response_model=AuthResponse)
def change_password(
    payload: ChangePasswordRequest,
    current_user: Person = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Change password while signed in. Returns a fresh token; other sessions are signed out."""
    user = db.query(Person).filter(Person.id == current_user.id).first()
    if not user or not user.password_hash or not verify_password(payload.current_password, user.password_hash):
        # 400 rather than 401 so the frontend keeps the session.
        raise HTTPException(status_code=400, detail="Your current password is incorrect")
    if payload.current_password == payload.new_password:
        raise HTTPException(status_code=400, detail="Choose a new password that is different from the current one")

    user.password_hash = hash_password(payload.new_password)
    user.token_version = (user.token_version or 0) + 1
    db.commit()
    db.refresh(user)
    return _auth_response(user)
