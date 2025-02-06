import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class DataSheet():
    def __init__(self,*,Name, Wounds, Toughness, Save, Invuln, Weapons):
        self.name = Name
        self.w = Wounds
        self.sv = Save
        self.inv = Invuln
        self.t = Toughness
        self.weapon = Weapons

class Weapon():
    def __init__(self,*, Attacks, Skill, Strength, Damage, ArmourPiercing):
        self.at = Attacks
        self.sk = Skill
        self.st = Strength
        self.d = Damage
        self.ap = ArmourPiercing

class unit():
    def __init__(self, ID, datasheet, number):
        self.id = ID
        self.model = datasheet
        self.number = number

lasgun = Weapon(Attacks=2,Skill=4,Strength=3,Damage=1,ArmourPiercing=0)
bolt_rifle = Weapon(Attacks=2,Skill=3,Strength=4,Damage=1,ArmourPiercing=1)
heavy_bolt_rifle = Weapon(Attacks=2,Skill=3,Strength=5,Damage=2,ArmourPiercing=1)
storm_bolter = Weapon(Attacks=4,Skill=3,Strength=4,Damage=1,ArmourPiercing=0)
guardian_spear = Weapon(Attacks=2,Skill=2,Strength=4,Damage=2,ArmourPiercing=1)
pulse_blaster = Weapon(Attacks=2,Skill=3,Strength=6,ArmourPiercing=1,Damage=1)

guards_man = DataSheet(Name="Guard", Wounds=1,Toughness=3,Save=5,Invuln=7,Weapons=lasgun)
tau_breacher = DataSheet(Name="Breacher",Wounds=1,Toughness=3,Save=4,Invuln=7,Weapons=pulse_blaster)
intercessor = DataSheet(Name="Intercessor",Wounds=2,Toughness=4,Save=3,Invuln=7,Weapons=bolt_rifle)
terminator = DataSheet(Name="Terminator",Wounds=3,Toughness=5,Save=2,Invuln=4,Weapons=storm_bolter)
heavy_intercessor = DataSheet(Name="H. Intercessor",Wounds=3,Toughness=6,Save=3,Invuln=7,Weapons=heavy_bolt_rifle)
custodian_guard = DataSheet(Name="Custodes",Wounds=3,Toughness=6,Save=2,Invuln=4,Weapons=guardian_spear)

def RollD6(n):
    return np.random.randint(6,size=n)+1

def RollHits(model, count):
    number_of_attacks = model.weapon.at*count
    hit_rolls = RollD6(number_of_attacks)
    number_of_successes = (hit_rolls>=model.weapon.sk).sum()
    return number_of_successes

def WoundRolls(strength,toughness,count):
    ratio = toughness/strength
    wound_rolls = RollD6(count)
    if ratio >= 2:
        number_of_successes = (wound_rolls>=6).sum()
    elif ratio > 1:
        number_of_successes = (wound_rolls>=5).sum()
    elif ratio == 1:
        number_of_successes = (wound_rolls>=4).sum()
    elif ratio < 1 and ratio > 0.5:
        number_of_successes = (wound_rolls>=3).sum()
    else:
        number_of_successes = (wound_rolls>=2).sum()
    return number_of_successes

def FailedSaveRolls(save,Invuln,count):
    if save > Invuln:
        return (RollD6(count)<Invuln).sum()
    else:
        return (RollD6(count)<save).sum()

def HandleDamage(wounds,failed_saves,damage):
    number_of_dead_models = 0
    for attacks in range(failed_saves):
        if damage >= wounds:
            number_of_dead_models += 1
        else:
            number_of_dead_models += damage / wounds
    return number_of_dead_models

def CombatRound(attacker, defender):
    '''
    attacker: unit    
    defender: unit    
    
    all models shoot, damage gets resolved, unit gets reduced
    if number of deadmoedels > unit size => end combat with victory'''
    number_of_hits = RollHits(attacker.model,np.ceil(attacker.number).astype(np.int32))
    number_of_wounds = WoundRolls(attacker.model.weapon.st,defender.model.t,number_of_hits)
    number_of_failed_saves = FailedSaveRolls(defender.model.sv + attacker.model.weapon.ap, defender.model.inv, number_of_wounds)
    number_of_dead_models = HandleDamage(defender.model.w,number_of_failed_saves,attacker.model.weapon.d)
    
    defender.number -= number_of_dead_models
    
    return attacker, defender

def SimulateRangedCombat(number_of_combats, model1, model2, points):
    lcm = np.lcm(points[0],points[1])
    wins = np.zeros(2) # unit1, unit2

    unit1 = unit(0,model1,(lcm//points[0]).astype(np.float32))
    unit2 = unit(1,model2,(lcm//points[1]).astype(np.float32))

    for i in range(number_of_combats):        
        unit1.number = (lcm//points[0]).astype(np.float32)      # initialise number of models 
        unit2.number = (lcm//[points[1]]).astype(np.float32)

        roll_off = np.random.randint(2) # decide intial attacker/defender of this fight

        if roll_off == 0:
            attacker = unit1
            defender = unit2
        else:
            attacker = unit2
            defender = unit1

        while True:
            attacker, defender = CombatRound(attacker, defender)
            if defender.number <= 0 :
                wins[attacker.id] += 1
                break
            else:
                temp = attacker         # if the defender hasn't been destroyed, swap sides and go again
                attacker = defender
                defender = temp
    return wins

def DegreeOfMistrust(k,n,p):
    return (k-p*n)**4 / (0.2 * ( n**5 * p**5 + ( n-n*p )**5))

def DegreeOfTrust(data,p):
    N = sum(data)
    prior = 0.5
    temp1 = prior * binom.pmf(data[0],N,p)
    temp2 = prior * DegreeOfMistrust(data[0],N,p)
    norm = temp1 + temp2
    return temp1/norm, temp2/norm

def BurnIn(burn_in_samples, number_of_particles, queue, points, cost, total_cost):
    for particle in range(number_of_particles):
        for i in range(len(queue)):
            for j in range(i+1,len(queue)):
                wins = SimulateRangedCombat(burn_in_samples,queue[i],queue[j],[points[particle,i],points[particle,j]])
                cost[particle,0,i,j] = np.abs(0.5 - wins[0]/burn_in_samples)**2
                cost[particle,0,i,-1] += wins[0]/burn_in_samples - 0.5
                cost[particle,0,i,-2] += cost[particle,0,i,j]

                cost[particle,1,i,j] =  wins[0]/burn_in_samples - 0.5
                cost[particle,1,j,i] = -cost[particle,1,i,j]

                cost[particle,0,j,i] = cost[particle,0,i,j]
                cost[particle,0,j,-1] += cost[particle,1,j,i]
                cost[particle,0,j,-2] += cost[particle,0,j,i]

        print(f"BurnIn: {100*(particle+1)/number_of_particles:.4} %",end="\r" )
        total_cost[particle] = (np.mean(cost[particle,0,:,-2]))

    return cost, total_cost

def OptimizationStep(nparticle, index,  number_of_samples, points, queue, cost):
    new_cost = np.zeros((2,len(queue)+2))

    for i in range(len(queue)):
        if index == i:
            new_cost[0,i] = 0
        else:
            wins = SimulateRangedCombat(number_of_samples,queue[index],queue[i],[points[nparticle,index], points[nparticle,i]])
            new_cost[0,i] = np.abs(wins[0]/number_of_samples - 0.5)**2
            new_cost[0,-1] += wins[0]/number_of_samples - 0.5
            new_cost[0,-2] += np.abs(wins[0]/number_of_samples - 0.5)**2
            new_cost[1,i] = wins[0]/number_of_samples - 0.5

    cost[nparticle,0,index,:] = new_cost[0,:]
    cost[nparticle,1,index,:] = new_cost[1,:]

    cost[nparticle,0,:,index] = new_cost[0,0:len(queue)]
    cost[nparticle,1,:,index] = -new_cost[1,0:len(queue)]

    for i in range(len(queue)):
        cost[nparticle,0,i,-1] = sum(cost[nparticle,1,i,0:len(queue)])
        cost[nparticle,0,i,-2] = sum(cost[nparticle,0,i,0:len(queue)])
    
    return cost

def Epoch(number_of_samples, number_of_particles, queue, points, cost, total_cost, best):
    for particle in range(number_of_particles):
        if total_cost[particle] <= total_cost[(particle - 1) % number_of_particles] and total_cost[particle] <= total_cost[(particle + 1) % number_of_particles]:   # if particle is locally the best, optimize by decreasing the largest contribution to the cost
            ind = np.argmax(cost[particle,0,:,-2])  # pick largest contributor to cost
            dir = cost[particle,0,ind,-1]           # optimize in direction that minimizes it

            beta = 0.25  # learning rate
            trust = DegreeOfTrust([(sum(cost[particle,1,0,0:len(queue)] + 0.5)*number_of_samples), len(queue)*number_of_samples - (sum(cost[particle,1,0,0:len(queue)] + 0.5)*number_of_samples)],0.5)[0] ## arrest the updates if we belief points are ok
            update = max(((points[particle,ind]*(1 + dir  * beta * (1-trust) ) + np.sign(dir)*(1 - trust) )  ).astype(np.int32),1)
            points[particle,ind] = update

            cost = OptimizationStep(particle, ind,  number_of_samples, points, queue, cost)

            total_cost[particle] = (np.mean(cost[particle,0,:,-2]))

        else:   # if particle is not local best, progress towards best known position
            dif = best - points[particle]   # direction to best known position
            ind = np.argmax(np.abs(dif))    # axis of largest difference
            
            phi = 0.75  # learning rate
            update = points[particle,ind] + np.round(phi*dif[ind]).astype(np.int32)
            points[particle,ind] = update
    
            cost = OptimizationStep(particle, ind,  number_of_samples, points, queue, cost)

            total_cost[particle] = (np.mean(cost[particle,0,:,-2]))

    return cost, total_cost

def Driver(number_of_epochs,        # Number of optimization steps
           number_burnin_samples,   # Number of samples used during the intial setup
           number_of_samples,       # number of samples used during optimization
           number_of_particles,     # number of particles in the swarm
           queue,                   # list of datasheets
           lims                     # upper and lower bound of random point initialisation
           ):
    
    points = np.random.randint(low=lims[0],high=lims[1],size=(number_of_particles,len(queue)))
    cost = np.zeros((number_of_particles,2,len(queue),len(queue)+2)) 
    total_cost = np.zeros(number_of_particles)
    
    cost_history = np.zeros((number_of_epochs+1,2))
    point_history = np.zeros((number_of_epochs+1,len(queue)))

    cost, total_cost = BurnIn(number_burnin_samples,number_of_particles,queue,points,cost,total_cost)

    best = points[np.argmin(total_cost)]
    cost_history[0,0] = np.min(total_cost)
    cost_history[0,1] = np.mean(total_cost)
    point_history[0] = best
    
    for ep in range(number_of_epochs):
        cost, total_cost = Epoch(number_of_samples, number_of_particles, queue, points, cost, total_cost, best)
    
        best = points[np.argmin(total_cost)]
    
        cost_history[ep+1,0] = np.min(total_cost)
        cost_history[ep+1,1] = np.mean(total_cost)
        point_history[ep+1] = best
        print(f"Progress: {100*(ep+1)/number_of_epochs:.4}%, ",f"Min: {cost_history[ep+1,0]:.4}, Mean: {cost_history[ep+1,1]:.4}  ", end="\r")
    
    ## Final Pass to get most accurate readings/
    # cost, total_cost = BurnIn(2000,number_of_particles,queue,points,cost,total_cost)

    return cost, total_cost, point_history, cost_history

## ---------------- Execution ---------------- ##

blo = 40 # limits for random point initialisation
bhi = 60 # large spread decreases learning speed

queue = [guards_man, intercessor, terminator, heavy_intercessor, custodian_guard] # 

cost, total_cost, point_history, cost_history = Driver(number_of_epochs=500,number_burnin_samples=100, number_of_samples=20, number_of_particles=20, queue=queue, lims=[blo,bhi])

c = cost[np.argmin(total_cost),1,:,0:len(queue)]
## ---------------- Plotting ---------------- ##

axticklabels = [model.name for model in queue]
axticklabels.insert(0,"")

plt.figure(figsize=(18,9))
ax1 = plt.subplot(2,1,2)
ax2 = plt.subplot(2,2,1)
ax3 = plt.subplot(2,2,2)

for i in range(len(queue)):
    ax1.plot(point_history[:,i],label=axticklabels[i+1])
ax1.grid()
ax1.legend()
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Points")

ax2.plot(cost_history[:,0],label="Min(Cost)")
ax2.plot(cost_history[:,1],label="Mean(Cost)")
ax2.grid()
ax2.legend()
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Cost")

im = ax3.imshow(c,clim=(-0.5,0.5),cmap="seismic")
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax, orientation='vertical')
ax3.set_xticklabels(axticklabels,rotation=25)
ax3.set_yticklabels(axticklabels)

plt.show()

# End
