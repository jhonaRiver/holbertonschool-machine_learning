#!/usr/bin/env python3
"""
module Poisson
"""


class Poisson:
    """
    class that represents a poisson distribution
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
            for value in self.data:
                sum += self.data[value]
            lambtha = sum / len(self.data)
            self.__lambtha = float(lambtha)

    def pmf(self, k):
        """
        calculates the value of the PMF for a given number of successes
        Args:
            k (int): number of successes

        Returns:
            float: PMF value for k
        """
        e = 2.7182818285
        k_fact = 1
        lambtha = self.__lambtha
        if type(k) != int:
            k = int(k)
        for i in range(1, k+1):
            k_fact *= i
        return (lambtha ** k / (e ** lambtha * k_fact))
