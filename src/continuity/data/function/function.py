import torch
from typing import List, Tuple, Self, Callable


class Function:
    """A class for creating and manipulating functions.

    This class allows for the encapsulation of arbitrary mathematical functions that operate on
    PyTorch tensors, providing an interface to perform basic arithmetic operations between functions
    such as addition, subtraction, multiplication, and division.

    Args:
        mapping: A callable that accepts exactly one argument, a torch.Tensor, and returns a torch.Tensor. This
        callable represents the mathematical function to be applied on a PyTorch tensor.

    Example:
        ```python
        f = Function(lambda x: x**2)
        g = Function(lambda x: x + 1)
        h = f + g
        x = torch.tensor(2.0)
        h(x)
        ```
        Output:
        ```shell
        tensor(7.)
        ```
    """

    def __init__(self, mapping: Callable):
        self.mapping = mapping

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the encapsulated mathematical function on a given tensor.

        Args:
            x: The input tensor to apply the function to.

        Returns:
            torch.Tensor: The result of applying the function to `x`.
        """
        return self.mapping(x)

    def __add__(self, other: Self) -> Self:
        """Creates a new Function representing the addition of this function with another.

        Args:
            other (Function): Another Function instance to add to this one.

        Returns:
            Function: A new Function instance representing the addition of the two functions.
        """
        return Function(mapping=lambda x: self.mapping(x) + other.mapping(x))

    def __sub__(self, other: Self) -> Self:
        """
        Creates a new Function representing the subtraction of another function from this one.

        Args:
            other (Function): Another Function instance to subtract from this one.

        Returns:
            Function: A new Function instance representing the subtraction of the two functions.
        """
        return Function(mapping=lambda x: self.mapping(x) - other.mapping(x))

    def __mul__(self, other: Self) -> Self:
        """Creates a new Function representing the multiplication of this function with another.

        Args:
            other (Function): Another Function instance to multiply with this one.

        Returns:
            Function: A new Function instance representing the multiplication of the two functions.
        """
        return Function(mapping=lambda x: self.mapping(x) * other.mapping(x))

    def __truediv__(self, other: Self) -> Self:
        """Creates a new Function representing the division of this function by another.

        Args:
            other: Another Function instance to divide this one by.

        Returns:
            Function: A new Function instance representing the division of the two functions.
        """
        return Function(mapping=lambda x: self.mapping(x) * other.mapping(x))


class ParameterizedFunction:
    """
    A class for creating and manipulating parameterized functions using PyTorch tensors.

    This class allows for the encapsulation of mathematical functions that not only operate on
    PyTorch tensors but are also parameterized by a set of parameters. It supports basic arithmetic
    operations (addition, subtraction, multiplication, and division) between instances, properly
    handling parameter updates.

    Args:
        mapping: A callable that accepts exactly two arguments, two torch.Tensors, and returns a torch.Tensor. This
            callable represents the mathematical parameterized function to be applied on a PyTorch tensor `x` with
            variable parameters `param`.
        n_parameters: The number of parameters that the `mapping` function expects.
        parameters_dtype: A list specifying the data types of the parameters, which aids in
            the construction and manipulation of parameters.


    Example:
        ```python
        mapping = lambda params, x: params[0] * x + params[1]
        pf = ParameterizedFunction(mapping, 2, [torch.float, torch.float])
        ```
        This creates a linear parameterized function `ax + b` where `a` and `b` are parameters.
    """

    def __init__(self, mapping: Callable, n_parameters: int, parameters_dtype: List):
        self.mapping = mapping
        self.n_parameters = n_parameters
        self.parameters_dtype = parameters_dtype

    def __call__(self, parameters: torch.Tensor) -> List[Function]:
        """Applies the parameterized function to the given parameters, returning a list of Function instances.

        Args:
            parameters: A tensor containing the parameters to be used with the mapping function.

        Returns:
            A list of Function instances, each parameterized by a single element from the
            `parameters` tensor.
        """
        return [Function(lambda x, a=param: self.mapping(a, x)) for param in parameters]

    def __get_operator_parameter_update(self, other: Self) -> Tuple[int, List]:
        """
        Private method to calculate the updated number of parameters and their data types after an arithmetic operation.

        Args:
            other: Another ParameterizedFunction instance to perform the operation with.

        Returns:
            A tuple containing the updated number of parameters and their data types. The order in which the parameters
                are updated follows self and then the others' parameters.
        """
        n_parameters = self.n_parameters + other.n_parameters
        parameters_dtype = self.parameters_dtype + other.parameters_dtype
        return n_parameters, parameters_dtype

    def __add__(self, other: Self) -> Self:
        """Creates a new ParameterisedFunction representing the addition of this parameterized function
        with another parameterized function.

        Args:
            other: Another ParameterizedFunction instance to add to this one.

        Returns:
            Function: A new Function instance representing the addition of the two functions.
        """
        n_parameters, parameters_dtype = self.__get_operator_parameter_update(other)
        return ParameterizedFunction(
            mapping=lambda a, x: self.mapping(a[: self.n_parameters], x)
            + other.mapping(a[self.n_parameters :], x),
            n_parameters=n_parameters,
            parameters_dtype=parameters_dtype,
        )

    def __sub__(self, other: Self) -> Self:
        """Creates a new ParameterisedFunction representing the subtraction of another parameterized function from this
        parameterized function.

        Args:
            other: Another ParameterizedFunction instance to subtract from this one.

        Returns:
            Function: A new Function instance representing the subtraction of the two functions.
        """

        n_parameters, parameters_dtype = self.__get_operator_parameter_update(other)
        return ParameterizedFunction(
            mapping=lambda a, x: self.mapping(a[: self.n_parameters], x)
            - other.mapping(a[self.n_parameters :], x),
            n_parameters=n_parameters,
            parameters_dtype=parameters_dtype,
        )

    def __mul__(self, other: Self) -> Self:
        """Creates a new Function representing the multiplication of this parameterized function with another.

        Args:
            other: Another ParameterizedFunction instance to multiply with this one.

        Returns:
            Function: A new ParameterizedFunction instance representing the multiplication of the two functions.
        """
        n_parameters, parameters_dtype = self.__get_operator_parameter_update(other)
        return ParameterizedFunction(
            mapping=lambda a, x: self.mapping(a[: self.n_parameters], x)
            * other.mapping(a[self.n_parameters :], x),
            n_parameters=n_parameters,
            parameters_dtype=parameters_dtype,
        )

    def __truediv__(self, other: Self) -> Self:
        """Creates a new ParameterizedFunction representing the division of this parameterized function by another.

        Args:
            other: Another ParameterizedFunction instance to divide this one by.

        Returns:
            Function: A new ParameterizedFunction instance representing the division of the two functions.
        """
        n_parameters, parameters_dtype = self.__get_operator_parameter_update(other)
        return ParameterizedFunction(
            mapping=lambda a, x: self.mapping(a[: self.n_parameters], x)
            / other.mapping(a[self.n_parameters :], x),
            n_parameters=n_parameters,
            parameters_dtype=parameters_dtype,
        )
