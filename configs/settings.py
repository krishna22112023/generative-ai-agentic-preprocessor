from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pyprojroot
from pydantic_settings import BaseSettings, SettingsConfigDict

from datetime import time
from typing import List


class PathInfo:
    """
    Base information class that defines core paths and environment settings.
    These paths are crucial for the application to locate resources and configurations.
    """

    HOME: Path = Path.home()
    BASE: Path = pyprojroot.find_root(pyprojroot.has_dir("config"))
    WORKSPACE: Path = BASE.parent.parent
    ENV = "dev"


# Load environment variables from .env file
load_dotenv(Path(PathInfo.BASE, ".env"))

class GeneralSettings(BaseSettings, PathInfo):

    model_config = SettingsConfigDict(case_sensitive=True)

