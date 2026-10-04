from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings shared by training and serving."""

    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    s3_bucket: str | None = Field(default=None, alias="S3_BUCKET")
    s3_prefix: str = Field(default="satellite-geospatial", alias="S3_PREFIX")

    # Classifier
    s3_model_key: str | None = Field(default=None, alias="S3_MODEL_KEY")
    model_path: Path = Field(default=Path("checkpoints/best_model.pt"), alias="MODEL_PATH")

    # Anomaly detector
    s3_autoencoder_key: str | None = Field(default=None, alias="S3_AUTOENCODER_KEY")
    autoencoder_path: Path = Field(default=Path("checkpoints/autoencoder_best.pt"), alias="AUTOENCODER_PATH")

    # Tree instance segmentation (Mask R-CNN)
    s3_segmentation_key: str | None = Field(default=None, alias="S3_SEGMENTATION_KEY")
    segmentation_path: Path = Field(default=Path("checkpoints/segmentation_best.pt"), alias="SEGMENTATION_PATH")

    image_size: int = Field(default=224, alias="IMAGE_SIZE")
    device: str = Field(default="cpu", alias="DEVICE")

    # Upload guards
    min_confidence: float = Field(default=0.6, alias="MIN_CONFIDENCE")
    max_upload_mb: int = Field(default=25, alias="MAX_UPLOAD_MB")
    max_pointcloud_points: int = Field(default=5_000_000, alias="MAX_POINTCLOUD_POINTS")
    # /predict/batch — caps how many images one request can score in a single
    # batched forward pass, protecting the same memory-constrained deployment
    # the /segment guards already account for.
    max_batch_size: int = Field(default=16, alias="MAX_BATCH_SIZE")

    # Rate limiting (per-client-IP, sliding 60s window). Heavy endpoints
    # (/segment, /pointcloud, /change-detect) get a lower cap since they are
    # the most memory/CPU expensive on the constrained free-tier deployment.
    rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")
    rate_limit_light_per_minute: int = Field(default=60, alias="RATE_LIMIT_LIGHT_PER_MINUTE")
    rate_limit_heavy_per_minute: int = Field(default=10, alias="RATE_LIMIT_HEAVY_PER_MINUTE")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


settings = Settings()
