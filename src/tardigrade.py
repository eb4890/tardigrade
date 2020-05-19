import math
import abc

class Distribution(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self):
        ...
    def set_num_samples(self, samples):
        self.num_samples = samples

    def __init__(self):
        self.num_samples = None


class UniformDistribution(Distribution):
    def sample(self):
        if (self.strategy=="random"):
            return random.uniform(self.lower, self.upper)
        elif (self.strategy=="coverage"):
            step = (self.upper - self.lower) / self.num_samples
            val = (step * self.samples_taken) + self.lower
            self.samples_taken+=1
            return val


    def __init__(self, upper, lower, strategy):
        self.upper = upper
        self.lower = lower
        self.strategy = strategy
        self.samples_taken = 0
        self.last_sample = None

class BernoulliDistribution(Distribution):
    def sample(self):
        if self.strategy == "random":
            val = random.uniform(0,1)
            if val < p:
                return 0
            else:
                return 1
        elif self.strategy=="coverage":
            num_zero_samples  = p * self.num_samples
            if self.samples_taken < num_zero_samples:
                return 0
            else :
                return 1

    def __init__(self, p, strategy):
        self.p = p
        self.samples_taken = 0
        self.strategy = strategy

class EquiProbableDistribution(Distribution):
    def sample(self):
        if self.strategy == "random":
            return random.choice(self.vals)
        elif self.strategy == "coverage":
            num_per_value = self.num_samples / self.vals.length
            val_num =  self.samples_taken / num_per_val
            self.samples_taken +=1
            return self.vals[math.floor(val_num)]

    def __init__(self, vals, strategy):
        self.vals = vals
        self.samples_taken = 0
        self.strategy = strategy


class SampleFrequency(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def should_resample(self, world):
        ...
    def force_resample(self):
        self.force_resample = True

    def __init__(self):
        self.force_resample = False

class Once(SampleFrequency):
    def shouldResample(self, world):
        if self.force_resample:
            self.force_resample=False
            return True
        return False

class Nticks(SampleFrequency):
    def shouldResample(self,world):
        if world.tick > self.ticks_per_sample + self.last_tick:
            self.last_tick = world.tick
            return True
        else:
            return False

    def __init__(self, ticks_per_sample):
        self.last_tick = 0
        self.ticks_per_sample = 0

class Model:
    def __init__(self, state, func):
        self.state = state
        self.func = func
    def run(self, world, state, time, time_diff):
       return self.func(world, state, time, time_diff)

class RandomVariable:
    def __init__(self,name, variable_type, frequency, distribution):
        self.name = name
        self.variable_type = variable_type
        self.frequency = frequency
        self.distribution = distribution
        self.value = None

    def getValue(self, world):
        if self.frequency.shouldResample(world) or self.value == None:
            self.value = self.distribution.sample()
        return self.value
    def set_num_samples(self, samples):
        self.distribution.set_samples()


class UncertainModel:
    def __init__(self):
        self.model_list = []
        self.random_variable = RandomVariable()
    def run(self, world,  state, time, time_diff):
        return self.model_list[
                self.random_variable.getValue(world)
            ].run(world, state, time, time_diff)

class State:
    def __init__(self, name, state_type, val):
        self.name = name
        self.type = state_type
        self.val= val

class World:
    def next(self, state, func):
        model = Model(state,func)
        if state in self.model_map:
            #Add random model here
            ...
        else:
            self.model_map[state] = model


    def add_state(self, name, state_type, val):
        self.state_map[name] = State(name, state_type, val)
    def add_variable(self, name, variable_type, distribution, sample_frequency):
        self.variable_map[name] = RandomVariable(name, variable_type, frequency,distribution)
    def copy_models(self, world, list_of_models_to_copy):
        ...
    def __init__(self):
        self.state_map = {}
        self.model_map ={}
        self.variable_map = {}
        self.history = []
        ...
    def load_state():
        ...
    def sample(self, num_samples):
        ...

    def probabilistic_model(self, state_name, distrubution, sampleFrequency):
        ...
    def simulate(self, steps):
        for var_name, var_obj in self.variable_map.items() :
            # We aren't sampling over lots of worlds
            var_obj.set_num_samples(1)

        for x in range(steps):
            for state_name,state_obj in self.state_map.items():
                next_state_value = self.model_map[state_name].run(world, state_obj.val, x, 1)
                state_obj.val = next_state_value
                print (state_name, next_state_value)



world = World()
world.add_state("light", "boolean", True )
world.next("light", lambda world , val, t, dt: True if t < 450 else False)
world.simulate(451)
