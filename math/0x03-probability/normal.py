#!/usr/bin/env python3
"""
module Normal class
"""


class Normal:
    """
    class that represents a normal distribution
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        class constructor
        Args:
            data (list, optional): list of data to be used to estimate the
                                   distribution. Defaults to None.
            mean (float, optional): mean of the distribution. Defaults to 0..
            stddev (float, optional): standard deviation of the distribution.
                                       Defaults to 1..
        """
        self.data = data
        self.mean = mean
        self.stddev = stddev

    @property
    def mean(self):
        """
        getter function
        Returns:
            float: mean of the distribution
        """
        return self.__mean

    @mean.setter
    def mean(self, mean):
        """
        setter function
        Args:
            mean (float): mean of the distribution

        Raises:
            TypeError: data must be a list
            ValueError: data must contain multiple values
        """
        if self.data is None:
            self.__mean = float(mean)
        else:
            if type(self.data) != list:
                raise TypeError("data must be a list")
            if len(self.data) < 2:
                raise ValueError("data must contain multiple values")
            sum = 0
            for i in range(0, len(self.data)):
                sum += self.data[i]
            mean = sum / len(self.data)
        self.__mean = float(mean)

    @property
    def stddev(self):
        """
        getter function
        Returns:
            float: standard deviation of the distribution
        """
        return self.__stddev

    @stddev.setter
    def stddev(self, stddev):
        """
        setter function
        Args:
            stddev (float): standard deviation of the distribution

        Raises:
            ValueError: stddev must be a positive value
            TypeError: data must be a list
            ValueError: data must contain multiple values
        """
        if stddev <= 0:
            raise ValueError("stddev must be a positive value")
        if self.data is None:
            self.__stddev = float(stddev)
        else:
            if type(self.data) != list:
                raise TypeError("data must be a list")
            if len(self.data) < 2:
                raise ValueError("data must contain multiple values")
            sum = 0
            for i in range(0, len(self.data)):
                sum += ((self.data[i] - self.__mean) ** 2)
            var = sum / len(self.data)
            stddev = var ** 0.5
        self.__stddev = float(stddev)

    def z_score(self, x):
        """
        calculates the z-score of a given x-value
        Args:
            x (float): x-value

        Returns:
            float: x-value of z
        """
        return ((x - self.__mean) / self.__stddev)

    def x_value(self, z):
        """
        calculates the x-value of a given z-score
        Args:
            z (float): z-score

        Returns:
            float: x-value of z
        """
        return ((z * self.__stddev) + self.__mean)
