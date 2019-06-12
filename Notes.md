# Game Theory
## Definition
Game Theory is a mathematical framework that models and analyzes situations where
multiple decision makers interact [36]. Using strategic decisions, each participant try to achieve
its objectives [37]


There are several types of games and corresponding representations. The two main distinctions
of game theory are between non-cooperative and cooperative game theory. In cooperative games,
the players request optimal concerted actions or reasonable cost/reward sharing rules that make
the coalitions stable and the players are allowed to communicate and to receive side payments in
order to act as one entity by improving their position in the game. Non-cooperative game theory
can be used to analyse decision making processes strategies of independent players which have
conflict preferences [33]. In these games, taking into consideration the actions of the other involved
players, each player optimizes its utility function automatically without coordination between the
strategic choices of each player. Nash equilibrium is one the most important solution concepts for
game theory [38]. In Nash equilibrium, no player can increase its revenues by changing unilaterally its
strategy, given that the actions of the other players are fixed. Therefore, a mutual optimal response
from all the players is achieved. Nash equilibrium in non-cooperative games is the optimum solution
when there is no leader-follower relationship and each player competes against the others in order to
maximize its utility function. In cooperative games, the strategies are achieved when the strategy set
can maximize the revenue of each player.

### Utility fonctions
#### Power Agent
The Power agent is a simple reflex agent which represents the PV system, the wind energy
system and the electrical energy demand. The agent monitors the power generation of the PVs and
wind turbine and calculates the energy balance of the system using Equation (1). The preference of
the agent was to maximize the energy production of the renewable sources:

`∆E = Ppv · t + Pwind · t − Pload · t`

#### Battery Agent
harging of the battery bank. The preference of this agent was to cover the load demand when the
RES could not provide enough power and to protect the battery bank which depends directly to the
battery SOC. The SOC should be kept between a minimum and maximum limit (Equation RES could not provide enough power and to protect the battery bank which depends directly to the
battery SOC. The SOC should be kept between a minimum and maximum limit (Equation (2)). In this
study, minimum depth of discharge (DOD) of the battery bank was set at 20% (Equation (3)) [52]:

`uBAT = 1 − (EBAT,max − EBAT)^2 / E BAT,max ^2`

### Game Theory for MAS


#### Sources
[1] A Game Theory Approach to Multi-Agent Decentralized Energy Management of Autonomous Polygeneration Microgrids