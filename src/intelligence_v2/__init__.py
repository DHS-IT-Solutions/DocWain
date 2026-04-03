"""DocWain Intelligence V2 — Profile Intelligence Builder and Alert Generator.

Provides pre-computed structured knowledge during document processing and
threshold-based alert detection across domains.
"""

from src.intelligence_v2.profile_builder import ProfileBuilder, ProfileIntelligence
from src.intelligence_v2.alert_generator import AlertGenerator, Alert

__all__ = [
    "ProfileBuilder",
    "ProfileIntelligence",
    "AlertGenerator",
    "Alert",
]
