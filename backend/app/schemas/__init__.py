"""Validated API request models."""

from __future__ import annotations

from datetime import date as Date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransactionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=500)
    merchant: str = Field(min_length=1, max_length=200)
    date: Date
    category: str = Field(min_length=1, max_length=100)
    amount: float = Field(gt=0, le=1_000_000_000)
    balance_after_transaction: float = Field(ge=-1_000_000_000_000, le=1_000_000_000_000)
    type: Literal["Income", "Expense"]
    subcategory: str = Field(default="Other", max_length=100)
    payment_mode: str = Field(default="Other", max_length=100)
    account_type: str = Field(default="Other", max_length=100)
    is_recurring: bool = False

    @field_validator("description", "merchant", "category", mode="before")
    @classmethod
    def trim_required_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class TransactionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = Field(default=None, min_length=1, max_length=500)
    merchant: str | None = Field(default=None, min_length=1, max_length=200)
    date: Date | None = None
    category: str | None = Field(default=None, min_length=1, max_length=100)
    amount: float | None = Field(default=None, gt=0, le=1_000_000_000)
    balance_after_transaction: float | None = Field(
        default=None,
        ge=-1_000_000_000_000,
        le=1_000_000_000_000,
    )
    type: Literal["Income", "Expense"] | None = None
    subcategory: str | None = Field(default=None, max_length=100)
    payment_mode: str | None = Field(default=None, max_length=100)
    account_type: str | None = Field(default=None, max_length=100)
    is_recurring: bool | None = None

    @field_validator(
        "description",
        "merchant",
        "category",
        "subcategory",
        "payment_mode",
        "account_type",
        mode="before",
    )
    @classmethod
    def trim_optional_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("*", mode="before")
    @classmethod
    def reject_null_update_values(cls, value: object) -> object:
        if value is None:
            raise ValueError("Update values cannot be null; omit unchanged fields.")
        return value


class BudgetCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1, max_length=100)
    amount: float = Field(gt=0, le=1_000_000_000)
    month: str | None = Field(default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$")

    @field_validator("category", mode="before")
    @classmethod
    def trim_category(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class BudgetUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str | None = Field(default=None, min_length=1, max_length=100)
    amount: float | None = Field(default=None, gt=0, le=1_000_000_000)
    month: str | None = Field(default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$")

    @field_validator("category", mode="before")
    @classmethod
    def trim_category(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("*", mode="before")
    @classmethod
    def reject_null_update_values(cls, value: object) -> object:
        if value is None:
            raise ValueError("Update values cannot be null; omit unchanged fields.")
        return value


class GoalCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=160)
    target: float = Field(gt=0, le=1_000_000_000)
    current: float = Field(default=0, ge=0, le=1_000_000_000)
    date: Date
    priority: str = Field(default="Medium", min_length=1, max_length=40)
    status: str = Field(default="active", min_length=1, max_length=40)

    @field_validator("name", "priority", "status", mode="before")
    @classmethod
    def trim_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class GoalUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=160)
    target: float | None = Field(default=None, gt=0, le=1_000_000_000)
    current: float | None = Field(default=None, ge=0, le=1_000_000_000)
    date: Date | None = None
    priority: str | None = Field(default=None, min_length=1, max_length=40)
    status: str | None = Field(default=None, min_length=1, max_length=40)

    @field_validator("name", "priority", "status", mode="before")
    @classmethod
    def trim_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("*", mode="before")
    @classmethod
    def reject_null_update_values(cls, value: object) -> object:
        if value is None:
            raise ValueError("Update values cannot be null; omit unchanged fields.")
        return value

class CategorizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[dict[str, str]] = Field(min_length=1, max_length=100)


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["forecast", "categorization", "anomaly"] = "forecast"
    transaction_id: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    merchant: str | None = Field(default=None, max_length=200)
