from dataclasses import dataclass


@dataclass
class Performance:
    fga: int
    fta: int
    percent_3pt: float
    percent_2pt: float
    plus_minus: float


@dataclass
class Player:
    name: float
    jersey_number: float
    health: float


@dataclass
class Team:
    players: list[Player]
    is_home: bool
