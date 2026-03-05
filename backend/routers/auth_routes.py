import logging
from typing import List, Literal, Optional

import auth_utils
from auth import (get_current_active_customer,
                  get_current_active_customer_unrestricted)
from database import (Customer, create_customer, get_customer_by_email,
                      get_subscription_plan_by_id, list_subscription_plans, plan_requires_payment,
                      plan_requires_sales_contact,
                      update_customer)
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from services.ec2_control import invoke_ec2_control
from services.email import send_welcome_email

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class UserCreate(BaseModel):
    email: str
    password: str
    name: str
    plan_id: str


class UserLogin(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: Optional[str] = None
    user: dict


class PasswordResetRequest(BaseModel):
    email: str


class PasswordReset(BaseModel):
    token: str
    new_password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class PlanResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    monthly_price_cents: int
    annual_price_cents: Optional[int] = None
    monthly_quota: int
    rate_limit_rpm: int
    concurrent_optimization_jobs: int
    batch_size_limit: int
    optimization_history_retention_days: int
    telemetry_retention_days: int
    audit_log_retention_days: int
    custom_canonical_mappings_limit: int
    max_api_keys: int
    features: List[str]
    is_active: bool
    is_public: bool = True
    plan_term: str = "monthly"
    monthly_discount_percent: int = 0
    yearly_discount_percent: int = 0


class ProfileUpdate(BaseModel):
    name: str
    phone_number: Optional[str] = None


class InfrastructureControlResponse(BaseModel):
    ok: bool
    action: str
    instance_state: Optional[str] = None
    details: dict


@router.post("/register", response_model=Token)
async def register(user_data: UserCreate):
    """Register a new user."""
    try:
        # Check if user already exists
        if get_customer_by_email(user_data.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        # Hash password
        password_hash = auth_utils.get_password_hash(user_data.password)

        # Validate and fetch the selected plan
        selected_plan = get_subscription_plan_by_id(
            user_data.plan_id,
            include_inactive=False,
            include_non_public=False,
        )
        if not selected_plan:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid plan: {user_data.plan_id}",
            )

        if plan_requires_sales_contact(selected_plan):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Selected plan requires contacting sales",
            )

        has_charge = plan_requires_payment(selected_plan)

        new_user = create_customer(
            name=user_data.name,
            email=user_data.email,
        )

        if not new_user:
            raise HTTPException(status_code=500, detail="Failed to create user")

        # Update with password hash and subscription info
        updated_user = update_customer(
            new_user.id,
            password_hash=password_hash,
            subscription_status="inactive" if has_charge else "active",
            subscription_tier=user_data.plan_id,
        )

        if not updated_user:
            raise HTTPException(status_code=500, detail="Failed to update user")

        try:
            send_welcome_email(
                to_email=updated_user.email,
                customer_name=updated_user.name or updated_user.email,
                plan=selected_plan,
            )
        except Exception:
            logger.exception("Failed to send welcome email to %s", updated_user.email)

        # Generate tokens
        access_token = auth_utils.create_access_token(
            data={
                "sub": updated_user.email,
                "id": updated_user.id,
                "role": updated_user.role,
            }
        )
        refresh_token = auth_utils.create_refresh_token(
            data={"sub": updated_user.email, "id": updated_user.id}
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "refresh_token": refresh_token,
            "user": {
                "id": updated_user.id,
                "email": updated_user.email,
                "name": updated_user.name,
                "phone_number": updated_user.phone_number,
                "role": updated_user.role,
                "subscription_status": updated_user.subscription_status,
                "subscription_tier": updated_user.subscription_tier,
                "quota_overage_bonus": updated_user.quota_overage_bonus,
                "created_at": updated_user.created_at,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login with email and password."""
    user = get_customer_by_email(form_data.username)
    if (
        not user
        or not user.password_hash
        or not auth_utils.verify_password(form_data.password, user.password_hash)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )

    access_token = auth_utils.create_access_token(
        data={"sub": user.email, "id": user.id, "role": user.role}
    )
    refresh_token = auth_utils.create_refresh_token(
        data={"sub": user.email, "id": user.id}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "refresh_token": refresh_token,
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "phone_number": user.phone_number,
            "role": user.role,
            "subscription_status": user.subscription_status,
            "subscription_tier": user.subscription_tier,
            "quota_overage_bonus": user.quota_overage_bonus,
            "created_at": user.created_at,
        },
    }


@router.post("/refresh", response_model=Token)
async def refresh_tokens(payload: RefreshTokenRequest):
    token_payload = auth_utils.decode_token(payload.refresh_token)
    if token_payload is None or token_payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    email: Optional[str] = token_payload.get("sub")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token payload",
        )

    user = get_customer_by_email(email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )

    access_token = auth_utils.create_access_token(
        data={"sub": user.email, "id": user.id, "role": user.role}
    )
    refresh_token = auth_utils.create_refresh_token(
        data={"sub": user.email, "id": user.id}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "refresh_token": refresh_token,
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "phone_number": user.phone_number,
            "role": user.role,
            "subscription_status": user.subscription_status,
            "subscription_tier": user.subscription_tier,
            "quota_overage_bonus": user.quota_overage_bonus,
            "created_at": user.created_at,
        },
    }


@router.get("/me")
async def read_users_me(
    current_user: Customer = Depends(get_current_active_customer_unrestricted),
):
    """Get current user profile."""
    # Don't return sensitive data like password hash
    return {
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.name,
        "phone_number": current_user.phone_number,
        "role": current_user.role,
        "subscription_status": current_user.subscription_status,
        "subscription_tier": current_user.subscription_tier,
        "quota_overage_bonus": current_user.quota_overage_bonus,
        "created_at": current_user.created_at,
    }


@router.put("/profile")
async def update_profile(
    profile_data: ProfileUpdate,
    current_user: Customer = Depends(get_current_active_customer),
):
    """Update user profile (name and phone number)."""
    try:
        if not profile_data.name.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Name cannot be empty",
            )

        updated_user = update_customer(
            current_user.id,
            name=profile_data.name,
            phone_number=profile_data.phone_number,
        )

        if not updated_user:
            raise HTTPException(status_code=500, detail="Failed to update profile")

        return {
            "id": updated_user.id,
            "email": updated_user.email,
            "name": updated_user.name,
            "phone_number": updated_user.phone_number,
            "role": updated_user.role,
            "subscription_status": updated_user.subscription_status,
            "subscription_tier": updated_user.subscription_tier,
            "quota_overage_bonus": updated_user.quota_overage_bonus,
            "created_at": updated_user.created_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Profile update failed: {str(e)}")


@router.get("/plans", response_model=List[PlanResponse])
async def list_plans():
    """List all active subscription plans (Public endpoint)."""
    plans = list_subscription_plans(include_inactive=False, include_non_public=False)
    plans.sort(
        key=lambda plan: (
            1 if plan_requires_sales_contact(plan) else 0,
            plan.monthly_price_cents if plan.monthly_price_cents >= 0 else 0,
            plan.monthly_quota if plan.monthly_quota >= 0 else 0,
        )
    )
    return [
        PlanResponse(
            id=p.id,
            name=p.name,
            description=p.description,
            monthly_price_cents=p.monthly_price_cents,
            annual_price_cents=p.annual_price_cents,
            monthly_quota=p.monthly_quota,
            rate_limit_rpm=p.rate_limit_rpm,
            concurrent_optimization_jobs=p.concurrent_optimization_jobs,
            batch_size_limit=p.batch_size_limit,
            optimization_history_retention_days=p.optimization_history_retention_days,
            telemetry_retention_days=p.telemetry_retention_days,
            audit_log_retention_days=p.audit_log_retention_days,
            custom_canonical_mappings_limit=p.custom_canonical_mappings_limit,
            max_api_keys=p.max_api_keys,
            features=p.features,
            is_active=p.is_active,
            is_public=p.is_public,
            plan_term=p.plan_term,
            monthly_discount_percent=p.monthly_discount_percent,
            yearly_discount_percent=p.yearly_discount_percent,
        )
        for p in plans
    ]


@router.get("/infrastructure/ec2/status", response_model=InfrastructureControlResponse)
async def get_ec2_status():
    """Public status endpoint used by login page controls."""
    return invoke_ec2_control("status")


@router.post(
    "/infrastructure/ec2/{action}",
    response_model=InfrastructureControlResponse,
)
async def control_ec2(action: Literal["start", "stop"]):
    """Public start/stop endpoint used by login page controls."""
    return invoke_ec2_control(action)
