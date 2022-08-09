#!/usr/bin/env python3
"""
module Exponential class
"""


class Exponential:
    """
    class that represents an exponential distribution
    """
    def __init__(self, data=None, lambtha=1.):
        """
        class constructor
        Args:
            data (list, optional): list of the data to be used to estimate
                                   the distribution. Defaults to None.
            lambtha (float, optional): expected number of occurences in a
                                       given time frame. Defaults to 1..
        """
        self.data = data
        self.lambtha = lambtha

    @property
    def lambtha(self):
        """
        getter function
        Returns:
            float: expected number of occurences in a given time frame
        """
        return self.__lambtha

    @lambtha.setter
    def lambtha(self, lambtha):
        """
        setter function
        Args:
            lambtha (float): expected number of occurences in a given time
                             frame

        Raises:
            ValueError: lambtha must be a positive value
            TypeError: data must be a list
            ValueError: data must contain multiple values
        """
        if lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        if self.data is None:
            self.__lambtha = float(lambtha)
        else:
            if type(self.data) != list:
                raise TypeError("data must be a list")
            if len(self.data) < 2:
                raise ValueError("data must contain multiple values")
            sum = 0
            for i in range(0, len(self.data)):
                sum += self.data[i]
            lambtha = 1 / (sum / len(self.data))
            self.__lambtha = float(lambtha)

    def pdf(self, x):
        """
        calculates the value of the PDF for a given time period
        Args:
            x (int): time period

        Returns:
            float: PDF value for x
        """
        e = 2.7182818285
        lambtha = self.__lambtha
        if x < 0:
            return 0
        return (lambtha * (e ** (-lambtha * x)))

    def cdf(self, x):
        """
        calculates the value of the CDF for a given time period
        Args:
            x (int): time period

        Returns:
            float: CDF value for x
        """
        e = 2.7182818285
        lambtha = self.__lambtha
        if x < 0:
            return 0
        return (1 - e ** (-lambtha * x))
