"""
Data distribution for none iid
"""

from typing import Any


class DataDistribution:
    
    #private
    __std_volume_mnist_lt = [
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

    #private
    __std_volume_mnist_lt_one_label = [
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

    #private
    __std_volume_mnist_balance = [
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

    #private dict to hold distribution
    __data_dict = {
        "mnist_lt": __std_volume_mnist_lt,
        "mnist_lt_one_label": __std_volume_mnist_lt_one_label,
        "mnist_data_volum_balance": __std_volume_mnist_balance,
    }

    # private use data list name
    __use_distribution_name: str = "mnist_lt"
    __use_volume_list: list[list[int]] = __std_volume_mnist_lt

    @staticmethod
    def distribution_name() -> str:
        """
        Get current dictribution name

        Returns:
            str: current distribution name
        """
        return DataDistribution.__use_distribution_name

    @staticmethod
    def get() -> list[list[int]]:
        """
        Get used distribution
        """
        return DataDistribution.__use_volume_list


    @staticmethod
    def add(name: str, volum_list: list[list[int]]):
        """
        Add another distribution
        """
        DataDistribution.__data_dict[name.strip()] = volum_list
        return


    @staticmethod
    def remove(name: str):
        """
        Remove distribution
        """
        n = name.strip()
        if DataDistribution.exists(n):
            del DataDistribution.__data_dict[n]
        return


    @staticmethod
    def exists(name: str) -> bool:
        """
        Check distribution exists
        """
        return name.strip() in DataDistribution.__data_dict


    @staticmethod
    def use(name = "mnist_lt", data_volume_list: list[list[int]]|None = None) -> list[list[int]]:
        """
        Use distribution pattern for data allocation.

        Args:
            distribution (str): Type of distribution ('mnist_lt' for long-tail, or others for user-defined).
            data_volume_list (array): data volume list if distribution pattern name not found

        Returns:
            list: A nested list where each sublist represents the data volume per class for a client.
        """

        n = name.strip()
        if DataDistribution.exists(n):
            DataDistribution.__use_distribution_name = n
            DataDistribution.__use_volume_list = DataDistribution.__data_dict[n]
        else:
            if data_volume_list is None:
                raise ValueError("'data_volume_list' can not be None.")
            DataDistribution.__use_volume_list = data_volume_list

        return DataDistribution.get()

    @staticmethod
    def parse_config(config_dict: dict):
        """
        Parse data distribution from config dict, sample yaml as below:

            # Data distribution define
            data_distribbution:

                # Data distribution valume name, predefined 'mnist_lt', 'mnist_lt_one_label', 'mnist_data_volum_balance' others use custom define
                use: mnist_lt

                # define custom data list
                custom_define:
                    custom: <define 'custom' data list>
                    name_xxx:  <define 'name_xxx' data list>
        """

        if "data_distribution" in config_dict:
            dict = config_dict["data_distribution"]
        else:
            dict = config_dict

        if "custom_define" in dict:
            for n in dict["custom_define"]:
                v = dict["custom_define"][n]
                DataDistribution.add(n, v)

        if "use" in dict:
            name = str(dict["use"]).strip()
            if DataDistribution.exists(name):
                DataDistribution.use(name)

        return