#!/usr/bin/env python3
"""
module Binomial class
"""


class Binomial:
    """
    class that represents a binomial distribution
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        class constructor
        Args:
            data (list, optional): list of the data to be used to estimate the
                                   distribution. Defaults to None.
            n (int, optional): number of Bernoulli trials. Defaults to 1.
            p (float, optional): probability of a "success". Defaults to 0.5.
        """
        self.data = data
        self.n = n
        self.p = p

    @property
    def n(self):
        """
        getter function
        Returns:
            int: number of Bernoulli trials
        """
        return self.__n

    @n.setter
    def n(self, n):
        """
        setter function
        Args:
            n (int): number of Bernoulli trials

        Raises:
            ValueError: n must be a positive value
            TypeError: data must be a list
            ValueError: data must contain multiple values
        """
        if n <= 0:
            raise ValueError("n must be a positive value")
        if self.data is None:
            self.__n = round(n)
        else:
            if type(self.data) != list:
                raise TypeError("data must be a list")
            if len(self.data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(self.data) / len(self.data)
            var = 0
            for i in range(len(self.data)):
                var += (self.data[i] - mean) ** 2
            variance = var / len(self.data)
            p = 1 - (variance / mean)
            n = mean / p
        self.__n = round(n)

    @property
    def p(self):
        """
        getter function
        Returns:
            float: probability of a "success"
        """
        return self.__p

    @p.setter
    def p(self, p):
        """
        setter function
        Args:
            p (float): probability of a "success"

        Raises:
            ValueError: p must be greater than 0 and less than 1
            TypeError: data must be a list
            ValueError: data must contain multiple values
        """
        if p <= 0 or p >= 1:
            raise ValueError("p must be greater than 0 and less than 1")
        if self.data is None:
            self.__p = float(p)
        else:
            if type(self.data) != list:
                raise TypeError("data must be a list")
            if len(self.data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(self.data) / len(self.data)
            p = mean / self.__n
        self.__p = float(p)

    def pmf(self, k):
        """
        calculates the value of the PMF for a given number of "successes"
        Args:
            k (int): number of "successes"

        Returns:
            float: PMF value for k
        """
        k = int(k)
        factor_k = 1
        factor_n = 1
        factor_c = 1
        if k < 0:
            return 0
        for i in range(1, k+1):
            factor_k *= i
        for i in range(1, self.__n+1):
            factor_n *= i
        for i in range(1, (self.__n - k)+1):
            factor_c *= i
        comb = factor_n / (factor_c * factor_k)
        prob = (self.__p ** k) * ((1 - self.__p) ** (self.__n - k))
        return (comb * prob)
