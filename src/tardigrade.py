"""Tardigrade is a module for experimenting with composable models for simulations.
The idea is that multiple people can define a model and they will work together well.
It also has ideas from deep uncertainty, in that it can easily represent uncertainty
between models as well.
"""

import math
import abc
import random


class Distribution(metaclass=abc.ABCMeta):
    """This is the Distribuion abstract base class. A Distribution is used by a random variable
    to reprsent the range and type of randomness that occurs"""

    @abc.abstractmethod
    def sample(self):
        """Sample picks a value from the distribution"""
        ...

    def set_num_samples(self, samples):
        """Set_num_samples sets the number of samples to be expect during the run
        This is usefull for coverage strategies, so that the samples can be evenly spaced.
        """
        self.num_samples = samples

    def __init__(self):
        self.num_samples = None


class UniformDistribution(Distribution):
    """UnifomDistribution selects a number in the range betweeen *upper* and *lower* with equal
    probability.
    There are two strategies that can be used when sampling. Random samples at random, coverage
    is used when yuu want to make sure you have even coverage over a space. Use this if you are only
    doing a few runs.
    """

    def sample(self):
        "Sample picks a value from the distribution"
        sample_val = None
        if self.strategy == "random":
            sample_val = random.uniform(self.lower, self.upper)
        elif self.strategy == "coverage":
            step = (self.upper - self.lower) / self.num_samples
            sample_val = (step * self.samples_taken) + self.lower
            self.samples_taken += 1
        return sample_val

    def __init__(self, upper, lower, strategy):
        super().__init__().__init__()
        self.upper = upper
        self.lower = lower
        self.strategy = strategy
        self.samples_taken = 0
        self.last_sample = None


class BernoulliDistribution(Distribution):
    """BernoulliDistrbution picks 0 with probability of p and 1
    with probability p-1"""

    def sample(self):
        "Sample picks a value from the distribution"
        sample_val = None
        if self.strategy == "random":
            val = random.uniform(0, 1)
            if val < self.p_val:
                sample_val = 0
            else:
                sample_val = 1
        elif self.strategy == "coverage":
            num_zero_samples = self.p_val * self.num_samples
            if self.samples_taken < num_zero_samples:
                sample_val = 0
            else:
                sample_val = 1
        else:
            pass
        return sample_val

    def __init__(self, p_val, strategy):
        super().__init__()
        self.p_val = p_val
        self.samples_taken = 0
        self.strategy = strategy


class EquiProbableDistribution(Distribution):
    """EquiprobableDistribution picks from a number of different
    values, equally likely. Like a dice roll."""

    def sample(self):
        "Sample picks a value from the distribution"
        sample_val = None
        if self.strategy == "random":
            sample_val = random.choice(self.vals)
        elif self.strategy == "coverage":
            num_per_value = self.num_samples / self.vals.length
            val_num = self.samples_taken / num_per_value
            self.samples_taken += 1
            sample_val = self.vals[math.floor(val_num)]
        else:
            pass
        return sample_val

    def __init__(self, vals, strategy):
        super().__init__()
        self.vals = vals
        self.samples_taken = 0
        self.strategy = strategy


class SampleFrequency(metaclass=abc.ABCMeta):
    """Sample frequency is the frequency at which a random variable is re-rolled
    This is the abstract base class"""

    @abc.abstractmethod
    def should_resample(self, world: World):
        """should_resample is called by the RandomVariable to determie if
        it is the right time to resample the variable"""
        ...

    def __init__(self):
        self.force_resample = False


class Once(SampleFrequency):

    """Once samples once during the run"""

    def should_resample(self, world: World):
        """should_resample is called by the RandomVariable to determie if
        it is the right time to resample the variable"""
        if self.force_resample:
            self.force_resample = False
            return True
        return False


class Nticks(SampleFrequency):
    """Nticks resamples the random variable once every n-ticks of the clock"""

    def should_resample(self, world: World):
        """should_resample is called by the RandomVariable to determine if
        it is the right time to resample the variable"""
        if world.tick > (self.ticks_per_sample + self.last_tick):
            self.last_tick = world.tick
            return True
        return False

    def __init__(self, ticks_per_sample):
        super().__init__()
        self.last_tick = 0
        self.ticks_per_sample = ticks_per_sample


class Model:
    """Models are what defines the next state. They take a function
    that is expected to take the world, the current state, the current time and
    the difference in time from the last world"""

    def __init__(self, state: Any, func: Callable[[World, Any, Int, Int], Any]):
        self.state = state
        self.func = func

    def run(self, world, state, time, time_diff):
        """Run runs the func with the variables"""
        return self.func(world, state, time, time_diff)


class RandomVariable:
    """RandomVariable brings together he frequency and distribution in order
    It also maintains knowledge of the current state of the random variable"""

    def __init__(
        self,
        name: str,
        variable_type: str,
        frequency: SampleFrequency,
        distribution: Distribution,
    ):
        self.name = name
        self.variable_type = variable_type
        self.frequency = frequency
        self.distribution = distribution
        self.value = None

    def get_value(self, world: World):
        """get_value gets the current value of the random variable. Sampling it if
        required"""
        if self.frequency.should_resample(world) or self.value is None:
            self.value = self.distribution.sample()
        return self.value

    def set_num_samples(self, samples: int) -> None:
        """Sets the number of samples for the run"""
        self.distribution.set_num_samples(samples)


class UncertainModel:
    """Uncertain Model has a number of different models for tbe behaviour. It uses a random
    variable to determine which one to add."""

    def __init__(self):
        self.model_list = []
        self.random_variable = None

    def add_model(self, model: Model) -> None:
        """Adds a new model to the possible models"""
        self.model_list.append(model)

    def add_random_variable(self, variable: RandomVariable) -> None:
        """Adds a new variable to control which model is picked"""
        self.random_variable = variable

    def run(self, world: World, state: Any, time: int, time_diff: int) -> Any:
        """Runs the model based on the random variable"""
        if self.random_varibale is None:
            Raise(
                Exception(
                    "Undefined random variable for uncertain model ",
                    self.model_list[0].name,
                )
            )
        return self.model_list[self.random_variable.get_value(world)].run(
            world, state, time, time_diff
        )


class State:
    """State holds the state"""

    def __init__(self, name, state_type, val):
        self.name = name
        self.type = state_type
        self.val = val

    def get_val(self) -> Any:
        """Gets the value of the state"""
        return self.val


class World:
    """World is the entire context for the simulation"""

    def next(self, state: State, func: Callable[[World, Any, Int, Int], Any]) -> None:
        """Next defines the function that updates the particular state over time"""
        model = Model(state, func)
        if state in self.model_map:
            # Add random model here
            ...
        else:
            self.model_map[state] = model

    def add_state(self, name, state_type, val):
        """Add state creates a new state of the world"""
        self.state_map[name] = State(name, state_type, val)

    def get_state(self, name):
        """Get state gets the current state"""
        return self.state_map[name].get_val()

    def add_variable(self, name, variable_type, distribution, sample_frequency):
        """Adds a random variable"""
        self.variable_map[name] = RandomVariable(
            name, variable_type, frequency, distribution
        )

    def copy_state_and_models(self, world):
        """Copies state and models from one world to another"""
        self.state_map.update(world.state_map)
        # TODO Need to do work in here if there are two models
        # with the same name to make an uncertain one
        self.model_map.update(world.model_map)

    def __init__(self):
        self.state_map = {}
        self.model_map = {}
        self.variable_map = {}
        self.history = []

    def sample(self, num_samples):
        ...

    def probabilistic_model(self, state_name, distrubution, sample_frequency):
        ...

    def simulate(self, steps):
        for var_name, var_obj in self.variable_map.items():
            # We aren't sampling over lots of worlds
            var_obj.set_num_samples(1)

        for x in range(steps):
            for state_name, state_obj in self.state_map.items():
                next_state_value = self.model_map[state_name].run(
                    self, state_obj.val, x, 1.0
                )
                state_obj.val = next_state_value
                print(state_name, next_state_value)


world1 = World()
world1.add_state("light", "boolean", True)
world1.next("light", lambda world, val, t, dt: True if t < 450 else False)

world2 = World()
world2.add_state("light", "boolean", True)
world2.add_state("carpos", "int", 0.0)
world2.next(
    "carpos",
    lambda world, val, t, dt: val + (dt / 6.0) if world.get_state("light") else val,
)


world3 = World()
world3.copy_state_and_models(world2)
world3.copy_state_and_models(world1)
world3.simulate(452)
print(world3.get_state("light"))
