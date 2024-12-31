1. Description

Three things I greatly enjoy are numerical optimization, data visualisation
and Warhammer 40k (40k). In 40k players build armies using miniatures according to
some point costs and limits. An idea I had for a while was to see if one could use
Monte Carlo to simulate fights and particle swarm optimization to adjust points.

The code picks a number of opposing models corresponding to their point costs and
the lowest common multiple of these costs, i.e. if a guardsman costs 5p and a
space marine costs 20, four guardsmen will be picked and one space marine.

Models in 40k have various attributes which make it more or less likely for them
to wound an enemy model or be wounded in turn. A typical "ranged combat" involves
A. rolling to hit,
B. rolling to wound
C. rolling armour saves,
D. applying damage.
Steps A.-C. are highly random in nature and will be simulated using Monte Carlo.

Over the course of a several such fights it should become clear if one side is
stronger than the other, which gives an indicator whether it is over- or under-
costed, as well as by how much. This can then be used to calculate the cost as the
the squared distance from the optimum:
|Nwins / Nfights - 0.5|^2

2. Classes
class DataSheet(): contains the characteristics of a given model
    def __init__(self,*,Name, Wounds, Toughness, Save, Invuln, Weapons):
        self.name = Name
        self.w = Wounds
        self.sv = Save
        self.inv = Invuln
        self.t = Toughness
        self.weapon = Weapons

class Weapon(): contains the characteristics of the weapons
    def __init__(self,*, Attacks, Skill, Strength, Damage, ArmourPiercing):
        self.at = Attacks
        self.sk = Skill
        self.st = Strength
        self.d = Damage
        self.ap = ArmourPiercing

class unit(): used during monte carlo to collect datasheet and model count into one structure
    def __init__(self, ID, datasheet, number):
        self.id = ID
        self.model = datasheet
        self.number = number


3. Functions
3.1. Monte Carlo
- RollD6(n): returns n random numbers between 1 and 6.

- Rollhits(model, count): returns number of successful hit rolls given the count
of the model and the weapon it is equipped with.

- WoundRoll(strength, toughness, count): returns number of successful wound roll
given the weapons strength and the targets toughness

- FailedSaveRolls(save,invuln,count): returns the number of failed saving throws
given the targets armour characteristic and the count of successful wound rolls.

- HandleDamage(wounds,failed_saves,damage): returns the number of destroyed models
given the targets number of wounds, the number of failed saves, and the weapon damage.

- CombatRound(attacker, defender): runs above functions in order.
returns attacker, defender with reduced numbers.

- SimulateRangedCombat(number_of_combats, model1, model2, points): runs CombatRound()
for as long as at least one model is alive. returns number of wins for each model

- DegreeOfTrust(data, p): returns bayesian degree of belief that Hypothesis that
"Point estimates are accurate such that model1 and model2 are well balanced" is true

3.2 Optimization
- BurnIn(burn_in_samples, number_of_particles, queue, points, cost, total_cost):
initialises cost matrix by letting each model fight each other model for each
particle in the swarm

- OptimizationStep(nparticle, index,  number_of_samples, points, queue, cost):
For each particle, for chosen model (index) fight against each other model. Update
and return cost matrix

- Epoch(number_of_samples, number_of_particles, queue, points, cost, total_cost, best):
If particle is locally best, pick largest contributor to its cost and minimize it
one step. Else, find globally best particle and move step in its direction along
one axis.

- Driver(number_of_epochs,number_burnin_samples,number_of_samples,number_of_particles,queue,lims):
initialises cost matrix, runs the optimization procedure.
returns cost, total cost, cost_history, point_history


4. Optimization procedure
The procedure I use is an adapted particle swarm optimizer. I chose PSO as there
is no explicit cost function that can be optimized and a meta-heuristic was necessary.
Standard PSO uses a number of particles which move in search-space according to some
innate velocity plus a velocity determined by a local and global best known position.

In the case of the present code the search space is the point costs of the models.
A particle is hence the cost-matrix evaluated at the given coordinates in point-space.
This matrix contains the win-ratios of each model against each other model. The
total_cost is estimated by the average win-ration of a model against each other model
using Monte Carlo.

This estimation is 1. very costly and 2. highly co-dependent, i.e.
increasing the win-rate of one model causes the win rate of all other models to be
decreased. Hence, I decided to only ever update a single model's point costs. This
allows optimization to be performed somewhat efficient while being as careful as
possible.

The direction is chosen in either of two ways: If the particle is the best among its
neighbours, it will pick the model which has the highest contribution to the total cost,
i.e. is the most unbalanced, and try to carefully improve its points.
If the particle is not locally the best, it will move in the direction of the best
known position in search-space.

This has the following advantages: 1. The locally best particles act as "scouts"
in the sense that they move semi freely across search space and can approach different
optima. The other particles will move to the best known position and thereby improving
the swarms average cost.

The optimization of the "scout-particles" involves Bayesian belief estimation of
how likely the results of the MC simulation are. There are a number of parameters
which enter the point update, which are dependent on the average win-rate of the
to-be-optimized model. These can lead to large-ish steps even if the average win-rate
lies is within the range that can be reasonably expected, given the number of
samples. The posterior of the Bayesian belief estimation is than used to slow
movement in search space until win-rates are re-evaluated next epoch or the particle
is no longer locally the best.

Such a belief estimation does not enter the optimization of the other particles.
Hence, once a particle is no longer locally the best as well as close to a reasonably
good point estimate, it will then move on the fastest route towards the best know
position in search space.

This has the following advantages: 1. "Scout particles" move freely but are less
likely to overshoot optimal position. 2. Once an optimum has been found all particles
will slowly converge towards it, switching between "scout" and normal roles.







#End
