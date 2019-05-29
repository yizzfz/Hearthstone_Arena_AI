import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import clr
import random
clr.AddReference(r'../SabberStoneDLL/SabberStoneCore')
clr.AddReference(r'../SabberStoneDLL/System.Memory')
clr.AddReference(r'../SabberStoneDLL/System.Runtime.CompilerServices.Unsafe')
clr.AddReference(r'../SabberStoneDLL/System.Buffers')
clr.AddReference(r'../SabberStoneDLL/System.Numerics.Vectors')

from System.Collections.Generic import List
from System import Int32
from SabberStoneCore.Model import Card, Cards, Game
from SabberStoneCore.Config import GameConfig
from SabberStoneCore.Enums import CardClass, State, PlayState
from SabberStoneCore.Tasks.PlayerTasks import *

heroclasses = [
CardClass.DRUID,
CardClass.HUNTER,
CardClass.MAGE,
CardClass.PALADIN,
CardClass.PRIEST,
CardClass.ROGUE,
CardClass.SHAMAN,
CardClass.WARLOCK,
CardClass.WARRIOR
]



class SabberEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.seed = None
        self.player1class = self._random_class()
        self.player2class = self._random_class()
        self.player1deck = self._random_deck(self.player1class)
        self.player2deck = self._random_deck(self.player1class)
        cfg = GameConfig()
        cfg.StartPlayer = -1
        cfg.Player1Name = "Player1"
        cfg.Player1HeroClass = self.player1class
        cfg.Player1Deck = self.player1deck
        cfg.Player2Name = "Player2"
        cfg.Player2HeroClass = self.player2class
        cfg.Player2Deck = self.player2deck
        cfg.FillDecks = False
        cfg.Shuffle = True
        cfg.SkipMulligan = False
        cfg.Logging = False
        self.cfg = cfg
        self.game = Game(self.cfg)
        game.StartGame()


    def _random_deck(self, class):
        cardset_neutral = Cards.FormatTypeClassCards(FormatType.FT_STANDARD)[CardClass.NEUTRAL]
        cardset_class = Cards.FormatTypeClassCards(FormatType.FT_STANDARD)[class]
        cardset = cardset_class + cardset_neutral

        deck = List[Card]()
        for i in range(30):
        	deck.Add(cardset[random.randint(0, len(cardset) - 1)])
        return deck



    def _random_class(self):
        r = random.randint(0, 8)
        return heroclasses[r]

    def step(self, action):

    def seed(self, seed=None):
        if seed != None:
            random.seed(seed)
        self.seed = seed
        return self.seed
