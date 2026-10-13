from pydantic import BaseModel, Field
from typing import Optional


class SignupRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=3, max_length=100)
    phone: Optional[str] = None
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=100)
    password: str


class AuthUser(BaseModel):
    id: int
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    role: str

    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUser


class SignupVerifyRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=100)
    code: str = Field(..., min_length=4, max_length=10)


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=100)


class ResetPasswordRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=100)
    code: str = Field(..., min_length=4, max_length=10)
    new_password: str = Field(..., min_length=8, max_length=128)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=8, max_length=128)


class AuthConfig(BaseModel):
    email_verification_required: bool
    password_reset_available: bool


class MessageResponse(BaseModel):
    message: str
