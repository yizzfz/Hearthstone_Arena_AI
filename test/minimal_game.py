# ref: https://github.com/HearthSim/SabberStone/blob/master/core-extensions/SabberStoneCoreAi/src/Program.cs
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


# create a card deck with 30 1/1 Wisps 
deck = List[Card]()
for i in range(30):
	deck.Add(Cards.FromName("Wisp"))

# game configuration
cfg = GameConfig()
cfg.StartPlayer = 1
cfg.Player1Name = "Player1"
cfg.Player1HeroClass = CardClass.PALADIN
cfg.Player1Deck = deck
cfg.Player2Name = "Player2"
cfg.Player2HeroClass = CardClass.MAGE
cfg.Player2Deck = deck
cfg.FillDecks = False
cfg.Shuffle = True
cfg.SkipMulligan = False
cfg.Logging = False
game = Game(cfg)

# game start
game.StartGame()
print('player 1 HP %d, player 2 HP %d' % (game.Player1.Hero.Health, game.Player1.Hero.Health))

# mulligan
game.Process(ChooseTask.Mulligan(game.Player1, List[int]()))
game.Process(ChooseTask.Mulligan(game.Player2, List[int]()))
game.MainReady()

# each player take random actions until game ends
while (game.State != State.COMPLETE):
	print('player 1 HP %d, player 2 HP %d' % (game.Player1.Hero.Health, game.Player1.Hero.Health))
	options = game.CurrentPlayer.Options()
	option = options[random.randint(0, len(options) - 1)]
	print(option.FullPrint())
	game.Process(option)
		
if game.Player1.PlayState == PlayState.WON:
	print('player 1 won')
else:
	print('player 2 won')