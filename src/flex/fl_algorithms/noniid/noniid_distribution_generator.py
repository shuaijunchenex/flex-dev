class NoniidDistributionGenerator:
    """
    Generate predefined or custom non-IID distributions for dataset partitioning.
    """
    MNIST_LT = [
        [592, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [592, 749, 0, 0, 0, 0, 0, 0, 0, 0],
        [592, 749, 744, 0, 0, 0, 0, 0, 0, 0],
        [592, 749, 744, 875, 0, 0, 0, 0, 0, 0],
        [592, 749, 745, 876, 973, 0, 0, 0, 0, 0],
        [592, 749, 745, 876, 973, 1084, 0, 0, 0, 0],
        [592, 749, 745, 876, 974, 1084, 1479, 0, 0, 0],
        [593, 749, 745, 876, 974, 1084, 1479, 2088, 0, 0],
        [593, 749, 745, 876, 974, 1084, 1480, 2088, 2925, 0],
        [593, 750, 745, 876, 974, 1085, 1480, 2089, 2926, 5949],
    ]

    MNIST_LT_ONE_LABEL = [
        [5920, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 6742, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 5958, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 6131, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5842, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 5421, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 5918, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 6265, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 5851, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 5949],
    ]

    MNIST_DATA_VOLUM_BALANCE = [
        [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
        [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
        [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
        [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
        [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
        [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
        [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
        [593, 674, 595, 613, 584, 542, 591, 626, 585, 594],
        [593, 674, 595, 613, 584, 542, 591, 626, 585, 594],
        [593, 674, 595, 613, 584, 542, 591, 626, 585, 594],
    ]

    @classmethod
    def distribution_generator(cls, distribution: str = "mnist_lt", data_volum_list=None):
        """
        Generates the distribution pattern for data allocation.

        Args:
            distribution (str): Type of distribution. One of:
                - 'mnist_lt'                  : long-tail distribution
                - 'mnist_data_volum_balance'  : balanced distribution
                - 'mnist_lt_one_label'        : one-label-per-client distribution
                - 'custom'                    : user-defined via data_volum_list
            data_volum_list (list): Custom data volume distribution, required if distribution='custom'.

        Returns:
            list: A nested list where each sublist represents the data volume per class for a client.
        """
        if distribution == "mnist_lt":
            return cls.MNIST_LT
        if distribution == "mnist_data_volum_balance":
            return cls.MNIST_DATA_VOLUM_BALANCE
        if distribution == "mnist_lt_one_label":
            return cls.MNIST_LT_ONE_LABEL
        if distribution == "custom":
            if data_volum_list is None:
                raise ValueError("Custom distribution requires 'data_volum_list'.")
            return data_volum_list
        raise ValueError(
            "Invalid distribution type. Choose one of "
            "'mnist_lt', 'mnist_data_volum_balance', 'mnist_lt_one_label', or 'custom'."
        )
