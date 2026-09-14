"""
Maps sign class names to the three official Indian sign categories:
    - regulatory (mandatory)    : must obey
    - cautionary (warning)      : warning of hazard
    - informatory               : information / direction

Covers the 85 YOLO-trained classes (SDI Indian Traffic Sign dataset).
Reference: MoRTH India Road Sign manual + Amrita/Megalingam et al. (2022).
"""
from __future__ import annotations

from typing import Dict


# ---- Regulatory (mandatory) signs ----
_REGULATORY = {
    "ALL_MOTOR_VEHICLE_PROHIBITED", "AXLE_LOAD_LIMIT",
    "BULLOCK_AND_HANDCART_PROHIBITED", "BULLOCK_PROHIBITED",
    "COMPULSARY_AHEAD", "COMPULSARY_AHEAD_OR_TURN_LEFT",
    "COMPULSARY_AHEAD_OR_TURN_RIGHT", "COMPULSARY_CYCLE_TRACK",
    "COMPULSARY_KEEP_LEFT", "COMPULSARY_KEEP_RIGHT",
    "COMPULSARY_MINIMUM_SPEED", "COMPULSARY_SOUND_HORN",
    "COMPULSARY_TURN_LEFT", "COMPULSARY_TURN_LEFT_AHEAD",
    "COMPULSARY_TURN_RIGHT", "COMPULSARY_TURN_RIGHT_AHEAD",
    "CYCLE_PROHIBITED", "GIVE_WAY", "HANDCART_PROHIBITED",
    "HEIGHT_LIMIT", "HORN_PROHIBITED", "LEFT_TURN_PROHIBITED",
    "LENGTH_LIMIT", "LOAD_LIMIT", "NO_ENTRY", "NO_PARKING",
    "NO_STOPPING_OR_STANDING", "OVERTAKING_PROHIBITED",
    "PEDESTRIAN_PROHIBITED", "PRIORITY_FOR_ONCOMING_VEHICLES",
    "RESTRICTION_ENDS", "RIGHT_TURN_PROHIBITED",
    "SPEED_LIMIT_5", "SPEED_LIMIT_15", "SPEED_LIMIT_20",
    "SPEED_LIMIT_30", "SPEED_LIMIT_40", "SPEED_LIMIT_50",
    "SPEED_LIMIT_60", "SPEED_LIMIT_70", "SPEED_LIMIT_80",
    "STOP", "STRAIGHT_PROHIBITED", "TONGA_PROHIBITED",
    "TRUCK_PROHIBITED", "U_TURN_PROHIBITED", "WIDTH_LIMIT",
}


# ---- Cautionary (warning) signs ----
_CAUTIONARY = {
    "BARRIER_AHEAD", "CATTLE", "CROSS_ROAD", "CYCLE_CROSSING",
    "DANGEROUS_DIP", "FALLING_ROCKS", "GAP_IN_MEDIAN",
    "GUARDED_LEVEL_CROSSING", "HUMP_OR_ROUGH_ROAD",
    "LEFT_HAIR_PIN_BEND", "LEFT_HAND_CURVE", "LEFT_REVERSE_BEND",
    "LOOSE_GRAVEL", "MEN_AT_WORK", "NARROW_BRIDGE",
    "NARROW_ROAD_AHEAD", "PEDESTRIAN_CROSSING",
    "QUAY_SIDE_OR_RIVER_BANK", "RIGHT_HAIR_PIN_BEND",
    "RIGHT_HAND_CURVE", "RIGHT_REVERSE_BEND", "ROAD_WIDENS_AHEAD",
    "SCHOOL_AHEAD", "SIDE_ROAD_LEFT", "SIDE_ROAD_RIGHT",
    "SLIPPERY_ROAD", "STAGGERED_INTERSECTION", "STEEP_ASCENT",
    "STEEP_DESCENT", "T_INTERSECTION", "UNGUARDED_LEVEL_CROSSING",
    "Y_INTERSECTION", "TRAFFIC_SIGNAL",
}


# ---- Informatory signs ----
_INFORMATORY = {
    "DIRECTION", "FERRY", "PASS_EITHER_SIDE", "ROUNDABOUT", "TURN_RIGHT",
}


def category_of(cls_name: str) -> str:
    """Return 'regulatory' | 'cautionary' | 'informatory' | 'unknown'."""
    if cls_name in _REGULATORY:
        return "regulatory"
    if cls_name in _CAUTIONARY:
        return "cautionary"
    if cls_name in _INFORMATORY:
        return "informatory"
    return "unknown"


def summarize_classes() -> Dict[str, int]:
    return {
        "regulatory": len(_REGULATORY),
        "cautionary": len(_CAUTIONARY),
        "informatory": len(_INFORMATORY),
        "total": len(_REGULATORY) + len(_CAUTIONARY) + len(_INFORMATORY),
    }


if __name__ == "__main__":
    counts = summarize_classes()
    for k, v in counts.items():
        print(f"{k}: {v}")