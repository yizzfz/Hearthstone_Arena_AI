# ref: https://github.com/HearthSim/SabberStone/blob/master/core-extensions/SabberStoneCoreAi/src/Program.cs
import os
import clr
import random
clr.AddReference(r'../SabberStoneDLL/SabberStoneCore')

from System.Collections.Generic import List
from System import Int32
from SabberStoneCore.Model import Cards, Game
from SabberStoneCore.Config import GameConfig
from SabberStoneCore.Enums import CardClass, State, PlayState
from SabberStoneCore.Tasks.PlayerTasks import *

cfg = GameConfig(
	StartPlayer = -1,
	Player1Name = "Player1",
	Player1HeroClass = CardClass.MAGE,
	Player1Deck = [
			Cards.FromName("Blessing of Might"),
			Cards.FromName("Blessing of Might"),
			Cards.FromName("Blessing of Might"),
			Cards.FromName("Blessing of Might")
			],
	Player2Name = "Player2",
	Player2HeroClass = CardClass.MAGE,
	Player2Deck = [
			Cards.FromName("Blessing of Might"),
			Cards.FromName("Blessing of Might"),
			Cards.FromName("Blessing of Might"),
			Cards.FromName("Blessing of Might")
			],
	FillDecks = False,
	Shuffle = True,
	SkipMulligan = False,
	Logging = False,
	History = False
)

game = Game(cfg)
game.StartGame()
print('player 1 HP %d, player 2 HP %d' % (game.Player1.Hero.Health, game.Player1.Hero.Health))

game.MainReady()

while (game.State != State.COMPLETE):
	print('player 1 HP %d, player 2 HP %d' % (game.Player1.Hero.Health, game.Player1.Hero.Health))
	# game.Process(HeroPowerTask.Any(game.CurrentPlayer))
	game.Process(EndTurnTask.Any(game.CurrentPlayer))

if game.Player1.PlayState == PlayState.WON:
	print('player 1 won')
else:
	print('player 2 won')