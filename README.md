# RegretMinDynamics

Simulation framework for analyzing regret minimization algorithms in multi-player games.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 3-Player Games (3D Cube Visualization)

Run all games with all algorithms:
```bash
python main.py
```

Run specific game:
```bash
python main.py --game PureCoordination
```

Run specific algorithm:
```bash
python main.py --algorithm OptFTRL
```

### 2-Player RPS Games (Simplex Visualization)

Run all RPS experiments with Fictitious Play:
```bash
python rps_experiments.py
```

Run specific RPS game:
```bash
python rps_experiments.py --game RPS
```

Run specific algorithm:
```bash
python rps_experiments.py --algorithm FictitiousPlay
```

## Supported Games

### 3-Player Games

**PureCoordination**
- Players receive reward only when they all coordinate on the same action pair
- Payoff structure: Symmetric coordination game
- Properties:
  - Multiple Nash equilibria
  - Requires all players to coordinate
  - Tests convergence to coordination

**CoordinationWithSpectator**
- Player 1 and Player 2 must coordinate to receive rewards
- Player 3 is a spectator who always receives 0 payoff regardless of actions
- Payoff structure: Asymmetric - only P1 and P2 receive non-zero payoffs
- Properties:
  - P1 and P2 must learn to coordinate
  - P3 has no incentive (always gets 0)
  - Tests algorithm behavior with inactive players

**MatchingPenniesWithTwist**
- Player 1 has a dominant strategy (action 1 always gives 0.1, action 0 gives 0)
- Players 2 and 3 play matching pennies against each other
- Payoff structure: Mixed-motive with dominant strategy
- Properties:
  - P1 should always play action 1
  - P2 and P3 have cyclic best responses
  - Tests convergence to dominant strategy equilibrium

**MatchingPenniesWithOutsideOption**
- Three-player game with competitive and cooperative elements
- Players can achieve mutual benefit or engage in conflict
- Payoff structure: Mixed-motive with coordination opportunities

**Game Intuition:**
The game has a special "golden outcome" [1, 1, 0] where all players get +1 (Pareto optimal). However, reaching this requires:
- P1 and P2 to both choose action 1 (coordinate)
- P3 to choose action 0 (the "outside option" - opting out of the conflict)

**Strategic Tension:**
- If P3 plays action 1, it creates a matching pennies dynamic where P1 wants to mismatch with P3
- If P3 plays action 0, P1 and P2 can coordinate on action 1 for mutual benefit
- P2 has an incentive to play action 1 when P1 plays 1, creating a coordination subgame
- The worst outcome [0, 1, 0] gives everyone -1 (mutual defection)

**Properties:**
- No pure strategy Nash equilibrium (cyclic best responses)
- Contains a Pareto optimal outcome that requires partial coordination
- Tests whether algorithms discover cooperative opportunities
- Player 3's choice critically affects whether conflict or cooperation emerges
- Complex strategic dependencies: each player's best response depends on both others

### 2-Player RPS Games

**RPS (Rock Paper Scissors)**
- Standard zero-sum game
- Each action beats one and loses to one
- Properties:
  - Unique Nash equilibrium: uniform mixing (1/3, 1/3, 1/3)
  - Zero-sum: players' interests are directly opposed
  - Symmetric game

**BiasedRPS**
- Rock receives a bonus (+0.2) when beating Scissors
- Non-zero-sum variant
- Properties:
  - Equilibrium shifts slightly toward Rock
  - Asymmetric payoffs
  - Tests algorithm response to biased rewards

**AsymmetricRPS**
- Player 1 gets standard payoffs, Player 2's payoffs are scaled (×0.7)
- Not zero-sum
- Properties:
  - Different incentive structures for each player
  - Equilibrium differs from standard RPS
  - Tests algorithm behavior in asymmetric settings

**NoisyRPS**
- Standard RPS with Gaussian noise (σ=0.1) added to payoffs
- Stochastic outcomes
- Properties:
  - Noisy feedback
  - Tests robustness to uncertainty
  - Equilibrium similar to standard RPS but harder to learn

## Supported Algorithms

### For 3-Player Games
- OptFTRL
- Hedge
- EXP3
- PGA
- RegretMatching
- RegretMatchingSoftmax
- BFTL_EXP3

### For 2-Player RPS Games
- FictitiousPlay
- SmoothFP_T0.1 (Smooth Fictitious Play with temperature 0.1)
- SmoothFP_T0.5 (Smooth Fictitious Play with temperature 0.5)

## Output

Generates GIF visualizations in the `visualizations/` directory:
- 3-player games: 3D cube showing strategy evolution
- 2-player games: Side-by-side simplex triangles for each player
