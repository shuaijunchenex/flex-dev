# How to use data distribution

## 1. Data Distribution class

- Predefine distribution name(case sensitive):
  - mnist_lt
  - mnist_lt_one_label
  - mnist_data_volum_balance
- Data Distribution is static class
- Method:
  - parse_config(config_dict): parse data distribution from config dictionary
  - use(name = "mnist_lt"): use distribution by name
  - get(): get distribution data list used
  - add(name: str, volume_list): add custom distribution data list
  - remove(name: str): remove distribution data list by name
  - exists(name: str): check if distribution name exists

## 2. Get Predefined distribution

Simple use DataDistribution.use() method like

```python
 d1 = DataDistribution.use("mnist_lt")
 print(f"Standard key: mnist_lt: \n{d1}\n")

 d2 = DataDistribution.use("mnist_lt_one_label")
 print(f"Standard key: mnist_lt_one_label: \n{d2}\n")

 d3 = DataDistribution.use("mnist_data_volum_balance")
 print(f"Standard key: mnist_data_volum_balance: \n{d3}\n")
```

## 3. Get  distribution from yaml file

In a yaml file define distribution section as follows

```yaml
#-----------------------------
# Data distribution section
#-----------------------------
data_distribution:

  # Use distribution name, predefined 'mnist_lt', 'mnist_lt_one_label', 'mnist_data_volum_balance',
  # others for custom distribution from 'custom_define'
  use: mnist_lt          

  # define custom data list SAMPLE
  custom_define:
    custom: [[5920, 0, 0, 0, 0, 0, 0, 0, 0, 0], ... , [0, 0, 0, 0, 0, 0, 0, 0, 0, 5949]]
    name_xxx: [[5920, 0, 0, 0, 0, 0, 0, 0, 0, 0], ... [0, 0, 0, 0, 0, 0, 0, 0, 0, 5949]]
```

Get distribution code like this

```python
# Load yaml config
config_file = './test_data/node_config_template.yaml'
config_dict = ConfigLoader.load(config_file)

# Parse distribution
DataDistribution.parse_config(config_dict)
# Get distribution
d = DataDistribution.get()
print(f"Config use distribution:  \n{d}\n")
```

Sample code at unit_test project, see unittest_data_distribution.py
