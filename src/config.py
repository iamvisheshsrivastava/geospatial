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
    max_upload_mb: int = Field(default=25, alias="MAX_UPLOAD_MB")
    max_pointcloud_points: int = Field(default=5_000_000, alias="MAX_POINTCLOUD_POINTS")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


settings = Settings()
