from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_utils import KeyValueMap, ConfigLoader, console

def test_normal_yaml():
    yaml_file_name = './test_data/node_config_template_client.yaml'

    # Load yaml file
    console.out(f"Test Ket-Value-Args from {yaml_file_name}")
    console.out(f"load yaml file: {yaml_file_name}")
    yaml = ConfigLoader.load(yaml_file_name)

    console.out("Build key-map from root '/' automaiticly, get all key-map.")
    console.out("------------- Begin ---------------")
    key_map = KeyValueMap.build_key_map(yaml, "/")
    for k, v in key_map.items():
        console.info(f"{k} = {v}")
    console.out("------------- End -----------------\n")

    v = key_map.get("data_loader/root", "Not found")

    console.out("Build key-map from yaml node '/foo1', get all key-map under '/foo1'.")
    console.out("------------- Begin ---------------")
    key_map = KeyValueMap.build_key_map(yaml, "/foo1")
    console.out("key-map from '/foo1'")
    for k, v in key_map.items():
        console.info(f"{k} = {v}")

    console.out("  \n'/foo1' Key-Value-Args dict")
    key_map = KeyValueMap(yaml["foo1"], key_map)
    for k, v in key_map._key_value_dict.items():
        console.info(f"  {k} = {v}")
    console.out("------------- End -----------------\n")

    console.out("Build key-map from yaml node '/dataset_loader', get all key-map under '/dateset'.")
    console.out("------------- Begin ---------------")
    console.out("key-map from '/dataset_loader'")
    key_map = KeyValueMap.build_key_map(yaml, "/dataset_loader")
    for k, v in key_map.items():
        console.info(f"{k} = {v}")

    console.out("  \n'/dataset_loader' Key-Value-Args dict")
    key_map = KeyValueMap(yaml["dataset_loader"], key_map)
    for k, v in key_map._key_value_dict.items():
        console.info(f"  {k} = {v}")
    console.out("------------- End -----------------\n")

    console.out("Custom build key-map.")
    console.out("------------- Begin ---------------")
    console.out("custom key-map:")
    key_map = { 
        "epochs": "training/epochs",
        "batch_size" : "training/batch_size",
        "shuffle": "training/shuffle",
        "lr": "training/lr" }
    for k, v in key_map.items():
        console.info(f"{k} = {v}")

    console.out("\nKey-Value-Args dict:")
    kv_args = KeyValueMap(yaml, key_map)
    for k, v in kv_args._key_value_dict.items():
        console.info(f"  {k} = {v}")

    console.out("------------- End -----------------\n")


    console.out("Custom build key-map from yaml branch node.")
    console.out("------------- Begin ---------------")
    console.out("Custom key-map:")
    key_map = { "lora": "*"}
    for k, v in key_map.items():
        console.info(f"{k} = {v}")

    console.out("\nKey-Value-Args dict")
    kv_args = KeyValueMap(yaml["lora"], key_map)
    for k, v in kv_args._key_value_dict.items():
        console.info(f"  {k} = {v}")
    console.out("------------- End -----------------\n")
    return

def test_yaml_get():
    yaml_file_name = './test_data/node_config_template_client.yaml'

    # Load yaml file
    console.out(f"Test Ket-Value-Args from {yaml_file_name}")
    console.out(f"load yaml file: {yaml_file_name}")
    yaml = ConfigLoader.load(yaml_file_name)

    console.out("Build key-map from root '/' automaiticly, get all key-map.")
    console.out("------------- Begin ---------------")
    kv_args = KeyValueMap(yaml)

    v = kv_args.get("dataset_loader/root", "not found")
    console.out(v)
    console.out("------------- End -----------------\n")
    return

def main():
    test_yaml_get()

    console.out("Normal test")
    test_normal_yaml()
    console.wait_any_key()

    console.out("Test set split char '~'")
    KeyValueMap.set_split_char("~")
    test_normal_yaml()
    console.wait_any_key()

    return

if __name__ == "__main__":
    main()
